# Changelog

Format based on [Keep a Changelog](https://keepachangelog.com/); versions follow SemVer.

## [1.0.0] - Unreleased

### Added
- `textsum` package with map-reduce summarization, BM25 retrieval, PDF Q&A and OCR.
- Streamlit app with Summarize, Ask your PDF and About tabs, length control and summary download.
- Unit tests, ruff configuration and a GitHub Actions workflow.
- `docs/ARCHITECTURE.md` design notes. Original project reports, slides and demo video are kept
  out of version control.
- `reference/` with the main source files from the four earlier repositories (API key redacted).
- README, architecture notes, contributing guide and `.env.example`.

### Changed
- Model is loaded from a Hugging Face id (`SUMMARIZER_MODEL`) instead of a hardcoded `D:\` path.
- Q&A no longer depends on LangChain/FAISS; the index is per session instead of a shared folder.

### Removed
- Hardcoded Google API key and committed FAISS files.
- Legacy `App.py`, `QA.py` and `Sumz.py`, and the commented-out dead code.

### Security
- The Google API key committed to the earlier repositories, and a Hugging Face token printed in the
  old project report PDF, must be revoked; both remain in those repositories' git history.
