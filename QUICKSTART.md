# Quick Start Guide - Symptom Checker Chatbot

**Status:** ✅ Project fully set up and ready to use!

---

## ✅ Current Status

**Your project is already configured with:**
- ✅ Environment: `symptomchecker` 
- ✅ Dependencies: All installed
- ✅ Dataset: In place (190MB)
- ✅ **Models: All 3 trained** (Random Forest, XGBoost, MLP)
- ✅ **Chatbot: Ready to use**

---

## 🚀 Running the Chatbot (YOU ARE HERE!)

Since all models are trained, simply run:

```bash
# Activate environment
conda activate symptomchecker

# Navigate to project
cd c:\Users\acer\Desktop\symptom-checker

# Start the chatbot
python chatbot.py
```

### Chat with Your AI Doctor! 💬

Describe symptoms naturally:
- "I have a headache and feel dizzy"
- "My stomach hurts and I feel nauseous"
- "Running a fever with a cough"

The chatbot will extract symptoms and predict possible diseases with confidence scores!

---

## 📊 Available Models

Choose which model to use:

```bash
# XGBoost (default, best performance)
python chatbot.py

# Random Forest
python chatbot.py --model random_forest

# MLP (Deep Learning)
python chatbot.py --model mlp
```

---

## 🎯 Chatbot Features

- **Natural Language**: Describe symptoms conversationally
- **Semantic Understanding**: Uses transformers to understand medical terms
- **Multiple Symptoms**: Collect symptoms through dialogue
- **Top-5 Predictions**: Ranked by confidence
- **Confidence Scores**: Visual progress bars
- **Medical Disclaimers**: Responsible AI usage

---

## 📖 Example Conversation

```
🏥 Welcome to the Disease Prediction Assistant!

🤖 Bot: Please describe your symptoms.

> I have a really bad headache and feel dizzy

🤖 Bot: I understand you're experiencing:
  • Headache
  • Dizziness

  Add more symptoms? (yes/no)

> yes, also tired

🤖 Bot: Added:
  • Fatigue

  Add more? (yes/no)

> no

🔍 DISEASE PREDICTION RESULTS

Symptoms: Headache, Dizziness, Fatigue

1. Migraine          ████████████████░░░░ 87.3%
2. Hypertension      ██████████░░░░░░░░░░ 65.2%
3. Vertigo           ████████░░░░░░░░░░░░ 58.4%
...

⚕️  Please consult a healthcare professional.
```

---

## 🛠️ Chatbot Commands

| Command | Action |
|---------|--------|
| `help` | Show available commands |
| `review` | List collected symptoms |
| `more/yes` | Add more symptoms |
| `predict/no/done` | Get predictions |
| `reset/new` | Start new consultation |
| `exit/quit` | Exit chatbot |

---

## 🔧 Optional: Monitor Training (Next Time)

If you ever retrain models, use the monitor:

```bash
# Terminal 1: Run training
python train.py

# Terminal 2: Run monitor
streamlit run training_monitor.py
```

Opens web UI at `http://localhost:8501` showing real-time progress!

---

## 📁 Project Structure

```
symptom-checker/
├── data/
│   └── disease_symptom_data.csv ✅
├── processed_data/              ✅ Generated
├── models/                      ✅ All 3 models trained
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── mlp.pth
├── chatbot.py                   ⭐ Run this!
├── symptom_extractor.py
├── conversation_manager.py
├── predict.py
├── train.py
└── training_monitor.py
```

---

## ⚠️ Important Notes

### Medical Disclaimer
This is for **informational purposes only**. Not a substitute for professional medical advice.

### One-Time Training
Models are already trained. You only need to run `train.py` again if:
- Models folder is deleted
- You update the dataset
- You change model parameters

### Keep It Simple
Just run `python chatbot.py` and start chatting!

---

## 💡 Tips for Best Results

1. **Be specific**: "headache" vs "severe headache on left side"
2. **Multiple symptoms**: More symptoms = better predictions
3. **Use common terms**: The AI understands medical synonyms
4. **Review before predicting**: Type `review` to check collected symptoms

---

## 🎉 You're All Set!

Everything is installed and trained. Just run:

```bash
python chatbot.py
```

And start your first symptom consultation! 🏥✨
