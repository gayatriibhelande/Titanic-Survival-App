import joblib
import pandas as pd
import streamlit as st

# The order of these columns must match the order used when the model was trained.
FEATURES = ["pclass", "sex", "age", "fare"]

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")


@st.cache_resource
def load_model():
    """Load the saved model once, so it is not reloaded on every click."""
    return joblib.load("model.joblib")


def validate(pclass, age, fare):
    """Return a list of problems with the input (empty list means it is valid)."""
    problems = []
    if pclass not in (1, 2, 3):
        problems.append("Ticket class must be 1, 2 or 3.")
    if not 0 < age <= 80:
        problems.append("Age must be between 1 and 80.")
    if not 0 <= fare <= 520:
        problems.append("Fare must be between 0 and 520.")
    return problems


st.title("🚢 Would this passenger have survived?")
st.write("An educational model trained on the Titanic passenger list.")

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox(
        "Ticket class", [1, 2, 3], index=2,
        help="1 = first class, 3 = third class",
    )
    sex = st.radio("Sex", ["female", "male"], horizontal=True)
with col2:
    age = st.slider("Age (years)", 1, 80, 30)
    fare = st.slider("Fare paid (£)", 0, 520, 30)

if st.button("Predict", type="primary"):
    problems = validate(pclass, age, fare)
    if problems:
        for p in problems:
            st.warning(p)
        st.stop()

    try:
        model = load_model()
    except FileNotFoundError:
        st.error("The model file (model.joblib) is missing. Train and save it first.")
        st.stop()
    except Exception:
        st.error("The model file could not be loaded. Please try again later.")
        st.stop()

    row = pd.DataFrame(
        [[pclass, 1 if sex == "female" else 0, age, fare]], columns=FEATURES
    )

    try:
        chance = model.predict_proba(row)[0][1]
    except Exception:
        st.error("Something went wrong while making the prediction.")
        st.stop()

    st.metric("Estimated chance of survival", f"{chance:.0%}")
    st.progress(float(chance))
    if chance >= 0.5:
        st.success("The model predicts this passenger would likely have survived.")
    else:
        st.info("The model predicts this passenger would likely not have survived.")

st.divider()
st.caption(
    "About this app: a decision tree (max depth 4) trained with scikit-learn on "
    "714 Titanic passengers using class, sex, age and fare. It reached about 80% "
    "accuracy on unseen test data. This is a learning exercise that describes a "
    "historical dataset. It is not advice, and no personal data is collected."
)
