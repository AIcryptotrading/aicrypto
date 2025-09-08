AI Paper Trading — v1 (Safe paper trading)

This package is a self-contained Streamlit app for paper-trading with an AI starter model.
Extract the ZIP and run the setup script for Windows, then run the app.

How to run (Windows):
1. Extract this ZIP to a folder, e.g. C:\aicrypto_paper_v1
2. Double-click setup.bat OR run in Command Prompt:
   setup.bat
3. After setup completes, run:
   venv\Scripts\activate
   streamlit run app.py

Notes:
- This is PAPER-TRADING only (no real orders sent).
- The AI is a starter (RandomForest) and a rule-based fallback. Train it using the UI.
- If installations of some heavy libraries fail on Windows, install them manually.
