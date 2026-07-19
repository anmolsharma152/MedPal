# MedPal — status handoff

| Field | Value |
|-------|--------|
| **As of** | 2026-07-19 |
| **Branch** | `main` |
| **Product** | Neuro-symbolic clinical decision support experiment — Tabular ResNet risk + dual-RAG + logic overrides |

---

## What exists

| Piece | Notes |
|-------|--------|
| Streamlit / app entry | `app.py`, `server.py`, `rag_chat.py` |
| Models | Under `models/` (risk nets) |
| Dual-RAG + neuro-symbolic story | Documented in root README |
| Hackathon problem statement PDF | In repo (gitignored policy may vary) |

This is a **demo / research-shaped** clinical AI prototype — not a production medical device.

---

## Gaps / care

- [ ] Clarify deploy path (local Streamlit vs server)  
- [ ] Stronger safety disclaimers in UI  
- [ ] Separate from WellnessMate product surface unless deliberate merge  

---

## Resume

```bash
cd ~/Projects/MedPal
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# set LLM / embedding keys if required by rag_chat
python app.py   # or streamlit run / server.py per README
```

See [setup.md](./setup.md).
