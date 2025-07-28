#  Mental Health Prediction Model

##  Overview

This project focuses on building a machine learning model to predict whether an individual is likely to need mental health treatment based on personal and workplace-related features. The final model is deployed using a Streamlit web application for real-time predictions.



##  Dataset

The dataset consists of responses to a mental health survey and includes a mix of numeric and categorical features.

### Features Used in the Model:

* **Age**: Numerical age (18–100).
* **Gender**: Categorical — `female`, `male`, `trans`.
* **Family History**: Whether the individual has a family history of mental illness.
* **Work Interference**: Degree to which mental health affects work (`Never`, `Rarely`, `Sometimes`, `Often`, `Don't know`).
* **Ease of Leave**: How easy it is to take leave for mental health reasons.
* **Anonymity**: Whether anonymity is protected by the employer.
* **Mental Health Consequence**: Whether discussing mental health could have negative consequences.
* **Benefits**: Whether mental health benefits are provided by the employer.

Other columns in the dataset were removed after feature analysis and tuning.



##  Model Training

###  Steps Followed:

1. **Data Cleaning**

   * Removed outliers in age (e.g., `99999999999`, negatives).
   * Filtered valid ages between 18 and 100.
   * Handled missing/null entries.

2. **Encoding**

   * Encoded categorical features using consistent label encodings.

3. **Feature Selection**

   * Selected only the most relevant 8 features based on impact and correlation.

4. **Model Building**

   * Used **RandomForestClassifier**.
   * Performed **hyperparameter tuning** using both `GridSearchCV` and `RandomizedSearchCV`.

5. **Evaluation**

   * Evaluated on test set using accuracy, confusion matrix, and classification report.
   * Achieved \~80–82% accuracy.

6. **Model Saving**

   * Final trained model saved using `joblib` as `mental_health_model.pkl`.



##  Deployment with Streamlit

### Features:

* Simple UI for entering user responses.
* Model prediction for mental health treatment need.
* Uses `model.feature_names_in_` to ensure feature consistency.
* Handles invalid inputs or mismatches gracefully.

### To Run the App:

1.  Install dependencies:

   ```bash
   pip install streamlit pandas scikit-learn joblib
   ```

2.  Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

3.  Open the URL in your browser (usually `http://localhost:8501`).



##  Challenges and Fixes

| Problem                        | Solution                                                |
| ------------------------------ | ------------------------------------------------------- |
| **Categorical mapping errors** | Used consistent mappings for all select fields          |
| **Feature mismatch**           | Used `model.feature_names_in_` to match input order     |
| **Imbalanced predictions**     | Used selected features with high information gain       |
| **Overfitting/Underfitting**   | Tuned model via `RandomizedSearchCV` and `GridSearchCV` |

---

##  Future Enhancements

*  Add model explanation using SHAP or LIME.
*  Improve UI with visual insights or risk levels.
*  Test with real-world diverse data.
*  Experiment with deep learning or ensemble stacking.
