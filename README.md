# 🚢 Titanic Survival Predictor (Streamlit)

An interactive web app that estimates a passenger's chance of surviving the Titanic,
using a decision tree trained with scikit-learn.

**Live app:** _paste your Streamlit link here_

## How it works
- `train_model.py` trains a `DecisionTreeClassifier(max_depth=4)` on class, sex, age and fare
  (714 passengers, 80/20 stratified split, about 80% test accuracy vs a 59% baseline) and saves it with joblib.
- `app.py` loads `model.joblib` once (`@st.cache_resource`), collects inputs with
  selectbox / radio / sliders, validates them, and shows the survival probability with `st.metric`.
- Missing model files and invalid input show friendly messages instead of error traces.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
To retrain the model first: `pip install seaborn` then `python train_model.py`.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `model.joblib` | Saved trained model |
| `train_model.py` | Script that trains and saves the model |
| `requirements.txt` | Packages for deployment (scikit-learn is pinned to the version that saved the model) |

## Limitations
Small dataset, only four features, one train/test split. This is a learning exercise that
describes a historical dataset. It is not advice, and the app collects no personal data.
