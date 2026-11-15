# Reference: original code

This folder keeps the main source files from the four earlier repositories this project was built
from. It is **for reading only**. It is not imported by the app, not linted and not tested. The
maintained code is in [`textsum/`](../textsum) and [`app.py`](../app.py).

| File here | Original repository and path | Notes |
|---|---|---|
| `app_ocr_grammar.py` | `Text-Summarization-using-LLM/App.py`, `-LLMs/App.py`, `-Mini-Project-/app.py` | Identical in all three. Summarize, PDF/image to text (OCR), QA tab and a placeholder Grammar Check. |
| `app_summarize_chat.py` | `Text-Summarization-using-LLMs/app.py` | Summarize plus "Chat with PDF" using Gemini and FAISS. |
| `summarizer_pipeline.py` | `Text-Summarization-using-LLM/Sumz.py` | Original summarizer app using the `transformers` pipeline. Identical in `-LLMs`. |
| `qa_chat_pdf_v1.py` | `Text-Summarization-using-LLM/QA.py` | Gemini + FAISS PDF chat. Identical in `-LLMs`. |
| `qa_chat_pdf_v2.py` | `QA-Bot/QA.py` | Same as v1 with a different page header. |
| `requirements_qa_bot.txt` | `QA-Bot/requirements.txt` | Pins used by the QA bot. |
| `model_config/` | `Text-Summarization-using-LLM-Mini-Project-` | Model config files from the mini-project upload. |

## Not included

- FAISS index files (`index.faiss`, `index.pkl`): leftover test data.
- Tokenizer and weight files (`spiece.model`, `tokenizer.json`, `training_args.bin`): download the
  model from Hugging Face instead.
- Reports, slides and the demo video (kept out of git).
- Editor and virtual-environment leftovers (`pyvenv.cfg`, `settings.json`) and one-line READMEs.

## Secrets removed

The earlier repositories hardcoded a Google API key in these files. It has been replaced with
`<REDACTED-GOOGLE-API-KEY>` here. The original key, and a Hugging Face token printed in the old
project report, must be treated as compromised and revoked; they remain in the old repositories'
git history.
