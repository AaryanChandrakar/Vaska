# Training Monitor - Quick Start

## What This Does

`training_monitor.py` is a Streamlit web app that shows real-time training progress by monitoring the `models/` directory.

## Features

- ✅ Progress bar (0-100%)
- ✅ Individual model status (Random Forest, XGBoost, MLP)
- ✅ File sizes and completion times
- ✅ Auto-refresh every 5 seconds
- ✅ Visual alerts when complete

## How to Use

### 1. Install Streamlit (if not installed)

```bash
conda activate symptomchecker
python -m pip install streamlit
```

### 2. Run the Monitor

Open a **NEW** terminal (keep training running in the original terminal):

```bash
# Activate environment
conda activate symptomchecker

# Navigate to project
cd c:\Users\acer\Desktop\symptom-checker

# Run monitor
streamlit run training_monitor.py
```

### 3. View in Browser

The app will automatically open in your browser at:
```
http://localhost:8501
```

## What You'll See

```
🤖 ML Model Training Monitor
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Overall Progress: 1/3 Models
[████████████░░░░░░░░░░░░░░░░░░░░░░░░░░] 33%
⏳ Training in progress... 33% complete

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model Status

Random Forest          ✅ Complete       1453.9 MB
                                        Finished: 15:09:23

Xgboost               ⏳ Training...

Mlp                   ⏳ Training...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Notes

- The monitor checks for **completed model files** every 5 seconds
- It **cannot** show epoch numbers or validation loss (those are only in the training terminal)
- Once all models are complete, the app will show a success message and balloons 🎉

## Troubleshooting

**Error: "streamlit: command not found"**
```bash
python -m pip install streamlit
```

**Monitor shows 0% but training is running**
- This is normal if no models have finished yet
- Random Forest will finish first (about 15-20 minutes)

**Want to stop the monitor?**
- Press `Ctrl+C` in the terminal running streamlit
- Or close the browser tab and terminal
