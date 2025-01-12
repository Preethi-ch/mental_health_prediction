import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("C:/Users/chpre/Downloads/random_forest_model.pkl")

# Define the Streamlit app
def main():
    st.title("Mental Health Prediction App")
    st.write("Fill in the details below to predict mental health outcomes.")
    

    # Input fields for the features
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    self_employed = st.selectbox("Self-employed", ["Yes", "No"])
    family_history = st.selectbox("Family history of mental illness", ["Yes", "No"])
    treatment = st.selectbox("Currently undergoing treatment", ["Yes", "No"])
    work_interfere = st.selectbox("Work interference", ["Never", "Rarely", "Sometimes", "Often"], index=2)
    no_employees = st.selectbox("Number of employees", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
    remote_work = st.selectbox("Remote work", ["Yes", "No"])
    tech_company = st.selectbox("Tech company", ["Yes", "No"])
    benefits = st.selectbox("Benefits provided", ["Yes", "No", "Don't know"])
    care_options = st.selectbox("Care options available", ["Yes", "No", "Not sure"])
    wellness_program = st.selectbox("Wellness program", ["Yes", "No", "Don't know"])
    seek_help = st.selectbox("Seek help available", ["Yes", "No", "Don't know"])
    anonymity = st.selectbox("Anonymity guaranteed", ["Yes", "No", "Don't know"])
    leave = st.selectbox("Ease of leave", ["Very difficult", "Somewhat difficult", "Don't know", "Somewhat easy", "Very easy"])
    mental_health_consequence = st.selectbox("Mental health consequence", ["Yes", "No", "Maybe"])
    phys_health_consequence = st.selectbox("Physical health consequence", ["Yes", "No", "Maybe"])
    coworkers = st.selectbox("Talk to coworkers", ["Yes", "No", "Some of them"])
    supervisor = st.selectbox("Talk to supervisor", ["Yes", "No", "Some of them"])
    mental_health_interview = st.selectbox("Mental health interview", ["Yes", "No"])
    phys_health_interview = st.selectbox("Physical health interview", ["Yes", "No"])
    mental_vs_physical = st.selectbox("Mental vs Physical Importance", ["Yes", "No", "Don't know"])
    obs_consequence = st.selectbox("Observed consequence", ["Yes", "No"])

    # Map categorical inputs to numerical values (ensure the same mapping as during training)
    mapping = {
        "Yes": 1, "No": 0, "Maybe": 2, "Don't know": 2,
        "Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3,
        "Very difficult": 0, "Somewhat difficult": 1, "Don't know": 2, 
        "Somewhat easy": 3, "Very easy": 4,
        "Male": 0, "Female": 1, "Other": 2,
        "1-5": 0, "6-25": 1, "26-100": 2, "100-500": 3, "500-1000": 4, "More than 1000": 5,
        "Some of them": 2
    }

    # Create a dictionary for the input
    input_data = {
        "Age": age,
        "Gender": mapping[gender],
        "self_employed": mapping[self_employed],
        "family_history": mapping[family_history],
        "treatment": mapping[treatment],
        "work_interfere": mapping[work_interfere],
        "no_employees": mapping[no_employees],
        "remote_work": mapping[remote_work],
        "tech_company": mapping[tech_company],
        "benefits": mapping[benefits],
        "care_options": mapping[care_options],
        "wellness_program": mapping[wellness_program],
        "seek_help": mapping[seek_help],
        "anonymity": mapping[anonymity],
        "leave": mapping[leave],
        "mental_health_consequence": mapping[mental_health_consequence],
        "phys_health_consequence": mapping[phys_health_consequence],
        "coworkers": mapping[coworkers],
        "supervisor": mapping[supervisor],
        "mental_health_interview": mapping[mental_health_interview],
        "phys_health_interview": mapping[phys_health_interview],
        "mental_vs_physical": mapping[mental_vs_physical],
        "obs_consequence": mapping[obs_consequence]
    }

    # Convert input dictionary to DataFrame with correct column order
    input_df = pd.DataFrame([input_data], columns=model.feature_names_in_)
    

    # Prediction
    if st.button("Predict"):
        try:
            prediction = model.predict(input_df)[0]
            if prediction == 1:
                st.success("The individual is predicted to have mental health issues.")
            else:
                st.success("The individual is predicted to NOT have mental health issues.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.write("Ensure feature names and values are consistent.")

# Run the app
if __name__ == "__main__":
    main()
