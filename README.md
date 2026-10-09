# MediMate prototype

Multi-agent medication companion for elderly patients (SDG 3: Good Health and Well-being).
Individual project by Syed Murtuza Amanath Hussain, MJCET, for the AICTE | IBM SkillsBuild
Machine Learning and Applied AI Internship 2026.

## Run it
Double-click `index.html` (any modern browser). Nothing to install, no server needed.

Try this demo flow:
1. Press **+15 min** to see the Reminder Agent send reminders for the 08:00 doses.
2. Press **Taken** on the first dose, then **+1 hour**. The Adherence Agent's risk score decides
   how soon the Caregiver Agent alerts the family when the other dose is not confirmed.
3. Press **I will call now** in the caregiver inbox, then mark the dose as taken.
4. Press **Jump to 19:55** and advance to see a missed evening dose.
5. Press **Finish day** to add the day to history and see the Doctor-Summary Agent report.

## How it works
- **Orchestrator** (rule-based) loads the schedule and coordinates the agents each simulated 5 minutes.
- **Reminder Agent** sends reminders in English, Hindi or Telugu; frequency depends on risk.
- **Adherence Agent** logs doses and predicts missed-dose risk with a decision tree (scikit-learn).
- **Caregiver Agent** alerts the caregiver when a dose is overdue beyond a risk-based limit.
- **Refill Agent** warns when 5 or fewer days of tablets remain.
- **Doctor-Summary Agent** builds a 7-day adherence report (printable).

## Honest limits
- The risk model is trained on **synthetic** data generated from assumptions (see `train_model.py`).
  Test AUC is about 0.68. Accuracy at the default cut-off is about the same as always predicting
  "dose taken", so the model is used to rank risk, not to classify. A real pilot needs consented real data.
- No real WhatsApp, SMS or voice messages are sent. The clock is simulated.
- Hindi and Telugu messages are draft templates and need native-speaker review.
- MediMate only reminds and tracks. It does not diagnose or change prescriptions.

## Files
- `index.html` the app (model embedded)
- `index.src.html` app source with a model placeholder
- `train_model.py` generates synthetic data, trains and exports `model.json`
- `build.py` embeds `model.json` into `index.html`

## Retrain (optional)
```
pip install scikit-learn numpy
python train_model.py
python build.py
```

## Put it online for a review link (free)
Drag the folder containing `index.html` onto https://app.netlify.com/drop
