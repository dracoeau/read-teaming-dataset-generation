#!/usr/bin/env python3
"""
Generate an OWASP Top 10 for LLMs (2025) red-teaming dataset with DeepTeam
(the red-teaming library built on DeepEval), 100% offline.

  * The only LLM used is your local Ollama (must be on localhost).
  * A socket-level guard blocks every connection that is not to a loopback
    address, so your chatbot description and data cannot leave the machine,
    even if a library tries (telemetry, update checks, cloud upload...).
  * Telemetry, update checks and Confident AI upload are also switched off.

Usage:
    python generate_redteam_dataset.py --profile chatbot_profile.yaml
    python generate_redteam_dataset.py --profile chatbot_profile.yaml --check   # guard + Ollama test only
"""

# ---------------------------------------------------------------------------
# 1) Lock the process down BEFORE importing deepteam / deepeval
# ---------------------------------------------------------------------------
import os

for var, value in {
    "DEEPTEAM_TELEMETRY_OPT_OUT": "YES",
    "DEEPEVAL_TELEMETRY_OPT_OUT": "YES",
    "DEEPEVAL_UPDATE_WARNING_OPT_OUT": "YES",  # deepteam's PyPI version check
    "DEEPEVAL_UPDATE_WARNING_OPT_IN": "0",     # deepeval's PyPI version check
    "ERROR_REPORTING": "NO",
    "NO_PROXY": "localhost,127.0.0.1,::1",
    "no_proxy": "localhost,127.0.0.1,::1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}.items():
    os.environ[var] = value

# No cloud credentials and no proxies in this process: nothing to fall back on.
for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
            "CONFIDENT_API_KEY", "DEEPEVAL_API_KEY", "AZURE_OPENAI_API_KEY",
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(var, None)

import ipaddress
import socket


class NetworkBlocked(ConnectionRefusedError):
    """Raised when anything tries to reach a non-loopback address."""


def _is_loopback(host) -> bool:
    if host is None:
        return True  # getaddrinfo(None, ...) = local bind
    if isinstance(host, bytes):
        host = host.decode()
    host = host.strip("[]").split("%")[0]
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False  # any other hostname = would need DNS = blocked


_orig_getaddrinfo = socket.getaddrinfo
_orig_connect = socket.socket.connect
_orig_connect_ex = socket.socket.connect_ex


def _guarded_getaddrinfo(host, *args, **kwargs):
    if not _is_loopback(host):
        raise NetworkBlocked(f"[offline guard] blocked DNS lookup for {host!r}")
    return _orig_getaddrinfo(host, *args, **kwargs)


def _check(sock, address):
    if sock.family in (socket.AF_INET, socket.AF_INET6) and not _is_loopback(address[0]):
        raise NetworkBlocked(f"[offline guard] blocked connection to {address!r}")


def _guarded_connect(self, address):
    _check(self, address)
    return _orig_connect(self, address)


def _guarded_connect_ex(self, address):
    _check(self, address)
    return _orig_connect_ex(self, address)


socket.getaddrinfo = _guarded_getaddrinfo
socket.socket.connect = _guarded_connect
socket.socket.connect_ex = _guarded_connect_ex

# ---------------------------------------------------------------------------
# 2) Now it is safe to import the libraries
# ---------------------------------------------------------------------------
import argparse
import csv
import json
import sys
from datetime import datetime
from urllib.parse import urlparse

import yaml
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.models import OllamaModel
from deepteam.attacks.attack_engine import AttackEngine
from deepteam.attacks.attack_simulator.attack_simulator import AttackSimulator
from deepteam.attacks.multi_turn import BaseMultiTurnAttack
from deepteam.frameworks import OWASPTop10
from deepteam.vulnerabilities import CustomVulnerability

OWASP_NAMES = {
    "LLM_01": "Prompt Injection",
    "LLM_02": "Sensitive Information Disclosure",
    "LLM_03": "Supply Chain",
    "LLM_04": "Data and Model Poisoning",
    "LLM_05": "Improper Output Handling",
    "LLM_06": "Excessive Agency",
    "LLM_07": "System Prompt Leakage",
    "LLM_08": "Vector and Embedding Weaknesses",
    "LLM_09": "Misinformation",
    "LLM_10": "Unbounded Consumption",
}


def load_profile(path):
    with open(path, encoding="utf-8") as f:
        p = yaml.safe_load(f)
    for key in ("ollama", "purpose"):
        if key not in p:
            sys.exit(f"Profile is missing '{key}'.")
    return p


def make_local_model(cfg):
    base_url = cfg.get("base_url", "http://localhost:11434")
    host = urlparse(base_url).hostname
    if not _is_loopback(host):
        sys.exit(f"Refusing to use Ollama at {base_url}: only localhost is allowed.")
    return OllamaModel(model=cfg["model"], base_url=base_url,
                       temperature=cfg.get("temperature", 0.7))


def self_test(model):
    print("• Offline guard: ", end="")
    try:
        socket.create_connection(("pypi.org", 443), timeout=3)
        sys.exit("FAILED — an external connection was possible. Stopping.")
    except NetworkBlocked:
        print("external network blocked ✓")
    print(f"• Local Ollama ({model.get_model_name()}): ", end="", flush=True)
    try:
        out = model.generate("Reply with the single word: ready")
        text = out[0] if isinstance(out, tuple) else out
        print(f"responded ✓ ({str(text).strip()[:40]!r})")
    except Exception as e:
        sys.exit(f"cannot reach it — is `ollama serve` running and the model pulled? ({e})")


def unique_attacks(attacks, include_multi_turn):
    seen, out = set(), []
    for a in attacks:
        if isinstance(a, BaseMultiTurnAttack) and not include_multi_turn:
            continue
        if a.get_name() not in seen:
            seen.add(a.get_name())
            out.append(a)
    return out


def bind_local(vulnerabilities, model):
    # Make sure no vulnerability keeps a cloud default model around.
    for v in vulnerabilities:
        v.simulator_model = model
        v.evaluation_model = model
    return vulnerabilities


def to_row(tc, category, category_name):
    vtype = tc.vulnerability_type
    return {
        "input": tc.input,
        "owasp_id": category,
        "owasp_category": category_name,
        "vulnerability": tc.vulnerability,
        "vulnerability_type": getattr(vtype, "value", str(vtype)),
        "attack_method": tc.attack_method,
        "turns": [t.model_dump() for t in tc.turns] if tc.turns else None,
        "error": tc.error,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="chatbot_profile.yaml")
    ap.add_argument("--check", action="store_true", help="only test the offline guard and Ollama")
    args = ap.parse_args()

    prof = load_profile(args.profile)
    model = make_local_model(prof["ollama"])
    self_test(model)
    if args.check:
        return

    purpose = " ".join(prof["purpose"].split())
    guidelines = list(prof.get("generation_guidelines", []))
    guidelines += [f"Context about the target system: {c}" for c in prof.get("domain_context", [])]

    engine = AttackEngine(
        simulator_model=model,
        variations=int(prof.get("variations_per_attack", 1)),
        generation_guidelines=guidelines or None,
        purpose=purpose,
    )
    n_per_type = int(prof.get("attacks_per_vulnerability_type", 1))
    run_all = bool(prof.get("run_all_attacks", False))
    include_mt = bool(prof.get("include_multi_turn", False))

    out_dir = prof.get("output_dir", "./redteam_output")
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    def simulate(vulns, attacks, cat_id, cat_name):
        sim = AttackSimulator(purpose=purpose, max_concurrent=1,
                              simulator_model=model, attack_engine=engine,
                              evaluation_model=model)
        tcs = sim.simulate(
            attacks_per_vulnerability_type=n_per_type,
            vulnerabilities=bind_local(vulns, model),
            attacks=attacks,
            ignore_errors=True,
            simulator_model=model,
            run_all_attacks=run_all,
        )
        new = [to_row(tc, cat_id, cat_name) for tc in tcs]
        rows.extend(new)
        ok = sum(1 for r in new if r["input"] and not r["error"])
        print(f"  → {ok}/{len(new)} attacks generated")

    # OWASP Top 10 — one category at a time so each row is tagged with its LLMxx id
    for cat in prof.get("owasp_categories", list(OWASP_NAMES)):
        print(f"\n=== {cat} {OWASP_NAMES[cat]} ===")
        fw = OWASPTop10(categories=[cat])
        simulate(fw.vulnerabilities, unique_attacks(fw.attacks, include_mt), cat, OWASP_NAMES[cat])

    # Chatbot-specific vulnerabilities, attacked with all single-turn OWASP methods
    custom = prof.get("custom_vulnerabilities") or []
    if custom:
        print("\n=== Custom (chatbot-specific) ===")
        vulns = [CustomVulnerability(name=c["name"], criteria=c["criteria"], types=c.get("types"),
                                     simulator_model=model, evaluation_model=model, attack_engine=engine)
                 for c in custom]
        simulate(vulns, unique_attacks(OWASPTop10().attacks, include_mt), "CUSTOM", "Chatbot-specific")

    # ------------------------------------------------------------------ save
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    good = [r for r in rows if r["input"] and not r["error"]]
    failed = [r for r in rows if not (r["input"] and not r["error"])]

    jsonl = os.path.join(out_dir, f"redteam_{stamp}.jsonl")
    with open(jsonl, "w", encoding="utf-8") as f:
        for r in good:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    csv_path = os.path.join(out_dir, f"redteam_{stamp}.csv")
    cols = ["owasp_id", "owasp_category", "vulnerability", "vulnerability_type", "attack_method", "input"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(good)

    # DeepEval goldens: load later with EvaluationDataset().add_goldens_from_json_file(...)
    ds = EvaluationDataset(goldens=[
        Golden(input=r["input"], additional_metadata={k: r[k] for k in cols if k != "input"})
        for r in good
    ])
    golden_path = ds.save_as(file_type="json", directory=out_dir, file_name=f"goldens_{stamp}")

    if failed:
        with open(os.path.join(out_dir, f"failed_{stamp}.jsonl"), "w", encoding="utf-8") as f:
            for r in failed:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    print(f"\nDone: {len(good)} attacks, {len(failed)} failed")
    print(f"  {jsonl}\n  {csv_path}\n  {golden_path}")


if __name__ == "__main__":
    main()
