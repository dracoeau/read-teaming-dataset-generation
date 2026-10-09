# Offline OWASP LLM Top 10 red-teaming dataset (DeepTeam + local Ollama)

DeepEval's red teaming now lives in **DeepTeam** (`pip install deepteam`), built on DeepEval.
This kit generates attacks tailored to your chatbot using **only** your local Ollama.

## 1. Install (one time, the only step that needs internet)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U deepteam ollama sentry-sdk pyyaml
ollama pull qwen2.5:14b        # or any local model you prefer
```
After this, you can unplug the network: nothing else is downloaded.

## 2. Describe your chatbot
Edit `chatbot_profile.yaml`: purpose, domain context, sensitive data types, custom vulnerabilities.
This file is only read locally and only sent to `localhost:11434`.

## 3. Run
```bash
ollama serve                                   # in another terminal, if not already running
python generate_redteam_dataset.py --check     # proves outside network is blocked + Ollama answers
python generate_redteam_dataset.py --profile chatbot_profile.yaml
```

Output in `./redteam_output/`:
- `redteam_*.jsonl` – one attack per line, tagged with OWASP id (LLM_01…LLM_10), vulnerability, type and attack method
- `redteam_*.csv` – same, for spreadsheets
- `goldens_*.json` – DeepEval goldens (`EvaluationDataset().add_goldens_from_json_file(...)`)
- `failed_*.jsonl` – generations the local model got wrong (bad JSON / refusals), if any

## How the data is kept local
1. **Socket guard** in the script: any connection or DNS lookup to a non-loopback address raises an error. This covers telemetry, update checks, Confident AI upload, or any default cloud model.
2. **Opt-outs** set before import: `DEEPTEAM_TELEMETRY_OPT_OUT`, `DEEPEVAL_TELEMETRY_OPT_OUT`, `DEEPEVAL_UPDATE_WARNING_OPT_OUT`, `ERROR_REPORTING=NO`.
3. Cloud API keys and proxy variables are removed from the process; Ollama URL must be localhost.
4. Don't run `deepteam login` / `deepeval login`.

For defence in depth, also run it on an air-gapped machine or behind an OS firewall rule that blocks all outbound traffic except 127.0.0.1.

## Language (e.g. French chatbot)
Set `language: "French"` in the profile. The script then:
1. adds a language rule at the start and end of every prompt sent to Ollama (a guideline alone is ignored by small models, because DeepTeam's own prompts are in English);
2. after generation, rewrites any attack still in English. Base64 / ROT-13 / Leetspeak attacks are decoded, rewritten and re-encoded. Rewritten rows have `"rewritten_to_language": true`.

`Multilingual` attacks are deliberately in another language (testing whether the bot leaks when asked in, say, Swahili). Set `keep_multilingual_attacks: false` to drop them.

## Notes
- Quality depends on the local model. 7–8B models often break JSON; 14B+ is more reliable. Safety-tuned models may refuse to write attacks; an uncensored local variant helps.
- Multi-turn jailbreaks (Crescendo, Tree, Linear…) need a live target to converse with, so they're off by default for a static dataset.
- Dataset size ≈ vulnerability types × `attacks_per_vulnerability_type` (× variations). With 1 per type it's ~190 attacks.
