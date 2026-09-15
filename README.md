# ⚡ Smart MCQ Solver: Academic Multiple-Choice Ranking with Transformers

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-ELECTRA--base-yellow.svg)](https://huggingface.co/Sandhya1528/smart-mcq-solver-app)
[![Kaggle Standing](https://img.shields.io/badge/Kaggle-Rank%20237%20%7C%20MAP%403%200.7589-20BEFF.svg)](https://www.kaggle.com/)

An end-to-end NLP project solving and ranking academic multiple-choice questions (physics, philosophy, and astrophysics) based on predicted probability using **Mean Average Precision at 3 (MAP@3)**.

---

## 📌 Highlights

* **Kaggle Leaderboard:** Achieved **Rank 237** with a test score of **0.75893** (blended ensemble).
* **Transfer Learning & Architectures:** 
  * Fine-tuned **ELECTRA-base** (`google/electra-base-discriminator`) via `AutoModelForMultipleChoice`.
  * Fine-tuned **DeBERTa-v3-small** (`microsoft/deberta-v3-small`) pairwise binary classification.
  * Custom Transformer encoder written from scratch in PyTorch.
* **Interactive Deployment:** Streamlit application backed by the fine-tuned ELECTRA weights hosted on Hugging Face Hub (`Sandhya1528/smart-mcq-solver-app`).

---

## 📊 Evaluation Summary

| Model | Validation Setup | Val MAP@3 | Role / Status |
| :--- | :--- | :---: | :--- |
| **TF-IDF + Logistic Regression** | 80/20 Stratified Hold-out | Baseline | Surface overlap baseline |
| **ELECTRA-base** | 5-fold GroupKFold | **0.6730** | Leakage-safe; deployed in Streamlit app |
| **DeBERTa-v3-small** | 85/15 Hold-out | **0.9556** | Pairwise sequence classification |
| **Transformer (From Scratch)** | 5-fold Stratified CV | **0.9947** | First-principles implementation |
| **Final Submission Ensemble** | Kaggle Private Test | **0.75893** | **Rank 237** (80% ELECTRA + 20% Scratch Transformer) |

---

## 📂 Repository Structure

```text
├── app/
│   ├── app.py             # Streamlit inference interface
│   └── runtime.txt        # Deployment environment spec
├── notebooks/
│   ├── 01_electra_training.ipynb
│   └── 02_deberta_v3_small.ipynb
├── docs/
│   └── smart_mcq_solver_report.pdf
├── requirements.txt
└── README.md
