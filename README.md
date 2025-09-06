# AI Crypto Trading - Full Project (Paper-trade)

This repository contains a Streamlit app + utilities for paper trading, backtesting,
supervised model starter, and placeholders for RL / HuggingFace integrations.

**Quick start (local)**:
1. Copy `config.yaml.sample` -> `config.yaml` and fill keys if needed.
2. Create venv and install:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run Streamlit:
   ```bash
   streamlit run app.py
   ```

**Notes**:
- This project is for educational / paper-trade use. Live trading requires secure key handling.
- RL training and large models require GPU/cloud resources; heavy packages are optional.
