import streamlit as st
import pandas as pd
import joblib

# Load the trained model (make sure this path is correct in your system)
model = joblib.load(r"C:\Users\chpre\OneDrive\Desktop\college\PROJECTS\ml_projects\menatl_health_prediction\mental_health_model.pkl")

# Streamlit app
def main():
    st.title(" Mental Health Prediction App")
    st.write("Fill in the details below to predict whether someone may need mental health treatment.")

    # Input fields — only selected, important features used during model training
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["female", "male", "trans"])
    family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
    work_interfere = st.selectbox("How often does your mental health interfere with work?", 
                                  ["Don't know", "Never", "Rarely", "Sometimes", "Often"], index=3)
    leave = st.selectbox("Ease of taking mental health leave", 
                         ["Don't know", "Somewhat difficult", "Somewhat easy", "Very difficult", "Very easy"])
    anonymity = st.selectbox("Is anonymity protected by employer?", ["Don't know", "No", "Yes"])
    mental_health_consequence = st.selectbox("Consequence of discussing mental health at work", ["Maybe", "No", "Yes"])
    benefits = st.selectbox("Does your employer provide mental health benefits?", ["Don't know", "No", "Yes"])

    # Mapping input values to match model encoding
    mapping = {
        "Yes": 1, "No": 0, "Maybe": 2, "Don't know": 2,
        "Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3,
        "Very difficult": 0, "Somewhat difficult": 1, "Somewhat easy": 3, "Very easy": 4,
        "female": 0, "male": 1, "trans": 2
    }

    # Input dictionary
    input_data = {
        "Age": age,
        "Gender": mapping[gender],
        "family_history": mapping[family_history],
        "work_interfere": mapping[work_interfere],
        "leave": mapping[leave],
        "anonymity": mapping[anonymity],
        "mental_health_consequence": mapping[mental_health_consequence],
        "benefits": mapping[benefits]
    }

    # Convert to DataFrame using same column order
    try:
        input_df = pd.DataFrame([input_data], columns=model.feature_names_in_)
    except Exception as e:
        st.error(f"⚠️ Feature mismatch: {e}")
        return

    # Predict
    if st.button("Predict"):
        try:
            prediction = model.predict(input_df)[0]
            if prediction == 1:
                st.error("⚠️ The model predicts that the individual MAY NEED mental health treatment.")
            else:
                st.success("✅ The model predicts that the individual is NOT likely to need mental health treatment.")
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Please verify all inputs and try again.")

# Run the app
if __name__ == "__main__":
    main()
