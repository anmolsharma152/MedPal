# MedPal — setup

Status: [STATUS.md](./STATUS.md). Feature narrative: root [README.md](../README.md).

```bash
cd ~/Projects/MedPal
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# .env for LLM/API keys if used — never commit
python app.py
# or: python server.py / streamlit run app.py  (match README)
```

Models live under `models/`. Utils under `utils/`.

**Disclaimer:** educational/research prototype — not for clinical diagnosis.
