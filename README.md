# Mental Health Prediction Model

## Overview
This project involves building a machine learning model to predict mental health outcomes based on a dataset containing various features related to personal, professional, and health-related factors. 
The model uses a Random Forest Classifier and is trained in a Google Colab.


## Dataset
The dataset used for training includes the following features:
- **Age**: Numeric value representing the individual's age.
- **Gender**: Categorical value representing the individual's gender.
- **Self-employed**: Indicates whether the individual is self-employed.
- **Family history**: Indicates whether the individual has a family history of mental illness.
- **Treatment**: Indicates whether the individual is currently undergoing treatment for mental health issues.
- **Work interference**: Indicates how mental health issues interfere with work.
- **Number of employees**: Indicates the size of the company the individual works for.
- **Remote work**: Indicates if the individual works remotely.
- **Tech company**: Indicates if the individual works in a tech company.
- **Benefits**: Indicates if the company provides mental health benefits.
- **Care options**: Indicates if mental health care options are available.
- **Wellness program**: Indicates if the company has a wellness program.
- **Seek help**: Indicates if resources to seek help are provided.
- **Anonymity**: Indicates if anonymity is guaranteed.
- **Leave**: Indicates the ease of taking leave for mental health.
- **Mental health consequence**: Indicates the potential consequence of discussing mental health issues.
- **Physical health consequence**: Indicates the potential consequence of discussing physical health issues.
- **Coworkers**: Indicates comfort in talking to coworkers about mental health.
- **Supervisor**: Indicates comfort in talking to supervisors about mental health.
- **Mental health interview**: Indicates if discussing mental health in an interview is acceptable.
- **Physical health interview**: Indicates if discussing physical health in an interview is acceptable.
- **Mental vs Physical**: Indicates whether mental health is prioritized as much as physical health.
- **Observed consequence**: Indicates if negative consequences are observed for discussing mental health.

The dataset contains categorical and numeric features. Categorical features are encoded into numeric values before training.


## Model Training

### Steps
1. **Data Preprocessing**:
   - Handle missing values.
   - Encode categorical variables using consistent mappings.
   - Normalize numeric variables if required.

2. **Model Selection**:
   - Use `RandomForestClassifier` as the primary model.
   - Perform hyperparameter tuning using `RandomizedSearchCV`.

3. **Training**:
   - Split the data into training and testing sets.
   - Train the Random Forest model on the training set.
   - Evaluate the model using accuracy and other metrics on the test set.

4. **Saving the Model**:
   - Save the trained model using `joblib` for deployment.


## Deployment

The model is deployed using a **Streamlit** application, which provides an interactive interface for users to input data and receive predictions. 
The app loads the saved model (`random_forest_model.pkl`) and performs real-time predictions.



## Streamlit Application

### Features
1. **User Input Form**:
   - Allows users to input data for all the features required by the model.
2. **Prediction**:
   - Processes the input data and returns a prediction (e.g., whether the individual is likely to have mental health issues).
3. **Error Handling**:
   - Handles version mismatches and feature mapping errors gracefully.

### Steps to Run the App
1. Install necessary libraries:
   ```bash
   pip install streamlit pandas joblib scikit-learn
   ```
2. Run the app using the command:
   ```bash
   streamlit run app.py
   ```
3. Open the provided URL in a web browser to interact with the app.



## Challenges and Solutions
1. **Feature Mapping Mismatch**:
   - Ensure consistent mapping of categorical variables between training and deployment environments.
2. **Version Mismatch**:
   - Align `scikit-learn` versions used during training and deployment.
   - Re-train the model if necessary to avoid compatibility issues.
3. **Model Compatibility**:
   - Use `model.feature_names_in_` to align the feature order in the deployment code.



## Future Enhancements
1. **Model Improvement**:
   - Experiment with additional models and ensemble techniques.
   - Use feature selection to improve model performance.
2. **Dataset Expansion**:
   - Include more diverse data to generalize predictions.
3. **UI Enhancements**:
   - Make the Streamlit app more interactive and visually appealing.
   - Add visualizations for predictions and model explanations.
