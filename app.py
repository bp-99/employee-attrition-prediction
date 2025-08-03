import pandas as pd
import numpy as np
import pickle
import streamlit as st

# Load models and scaler
with open('xgboost_model.pkl', 'rb') as f:
    attrition_model = pickle.load(f)

with open('rfc_model.pkl', 'rb') as f:
    performance_model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    le_dict = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# Define app structure
st.sidebar.title('Navigation')
page = st.sidebar.radio("Go to", ["🏠 Home", "Employee Attrition Predictor", "Performance Rating Predictor"])

# Get EXACT feature names from models (convert to lowercase for consistency)
attrition_features = [f.lower() for f in attrition_model.feature_names_in_]
performance_features = [f.lower() for f in performance_model.feature_names_in_]

# Common categorical inputs (using lowercase keys to match)
categorical_inputs = {
    'businesstravel': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
    'department': ['Sales', 'Research & Development', 'Human Resources'],
    'educationfield': ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'],
    'jobrole': ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director',
                'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'],
    'maritalstatus': ['Single', 'Married', 'Divorced'],
    'gender': ['Male', 'Female'],
    'overtime': ['Yes', 'No']
}

def collect_inputs(for_attrition=True):
    input_data = {}
    st.subheader("Enter Employee Information")
    cols = st.columns(3)
    i = 0
    
    # Use the correct feature list based on model
    features_to_show = attrition_features if for_attrition else performance_features

    for feature in features_to_show:
        col = cols[i % 3]
        with col:
            label = feature.replace('_', ' ').title()

            if feature in categorical_inputs:
                input_data[feature] = st.selectbox(label, categorical_inputs[feature], key=f"{feature}_{'att' if for_attrition else 'perf'}")
            elif feature == 'monthlyincome':
                input_data[feature] = st.number_input(label, min_value=1000, key=f"{feature}_{'att' if for_attrition else 'perf'}")
            elif feature == 'distancefromhome':
                input_data[feature] = st.number_input(label, min_value=0.0, key=f"{feature}_{'att' if for_attrition else 'perf'}")
            elif feature == 'performancerating' and for_attrition:
                input_data[feature] = st.selectbox(label, list(range(1, 5)), key="perf_rating_att")
            elif feature in ['education', 'environmentsatisfaction', 'jobinvolvement', 'jobsatisfaction',
                           'relationshipsatisfaction', 'stockoptionlevel', 'worklifebalance']:
                input_data[feature] = st.selectbox(label, list(range(1, 5)), key=f"{feature}_{'att' if for_attrition else 'perf'}")
            elif feature == 'joblevel':
                input_data[feature] = st.selectbox(label, list(range(1, 6)), key=f"{feature}_{'att' if for_attrition else 'perf'}")
            elif feature in ['numcompaniesworked', 'trainingtimeslastyear']:
                input_data[feature] = st.selectbox(label, list(range(0, 11)), key=f"{feature}_{'att' if for_attrition else 'perf'}")
            elif feature in ['totalworkingyears', 'yearsatcompany']:
                input_data[feature] = st.selectbox(label, list(range(0, 41)), key=f"{feature}_{'att' if for_attrition else 'perf'}")
            elif feature in ['yearsincurrentrole', 'yearssincelastpromotion', 'yearswithcurrmanager']:
                input_data[feature] = st.selectbox(label, list(range(0, 40)), key=f"{feature}_{'att' if for_attrition else 'perf'}")
            elif feature == 'percentsalaryhike':
                input_data[feature] = st.selectbox(label, list(range(0, 101)), key=f"{feature}_{'att' if for_attrition else 'perf'}")
            elif feature == 'age':
                input_data[feature] = st.number_input(label, min_value=18, max_value=65, key=f"{feature}_{'att' if for_attrition else 'perf'}")

        i += 1

    return input_data

def preprocess_input(input_dict, for_attrition=True):
    # Get the exact features the model expects (in original case)
    expected_features = attrition_model.feature_names_in_ if for_attrition else performance_model.feature_names_in_
    scaler_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else []

    # Build a row dictionary for DataFrame
    row = {}
    for feature in expected_features:
        feature_lower = feature.lower()
        if feature_lower in input_dict:
            if feature_lower in categorical_inputs:
                try:
                    row[feature] = [le_dict[feature].transform([input_dict[feature_lower]])[0]]
                except Exception as e:
                    st.error(f"Encoding error for {feature}: {e}")
                    raise
            else:
                row[feature] = [input_dict[feature_lower]]
        else:
            st.error(f"Missing required feature: {feature} (looked for {feature_lower} in inputs)")
            raise ValueError(f"Missing feature: {feature}")

    # Create DataFrame with one row
    df = pd.DataFrame(row)

    # Only scale the features the scaler was trained on
    if len(scaler_features) > 0:
        to_scale = df[scaler_features]
        scaled = scaler.transform(to_scale)
        scaled_df = pd.DataFrame(scaled, columns=scaler_features)
        # Replace scaled columns in df
        for col in scaler_features:
            df[col] = scaled_df[col]

    # Ensure columns are in the order expected by the model
    df = df[list(expected_features)]
    return df.values

# 🏠 Home Page
if page == '🏠 Home':
    st.title("Employee Attrition & Performance Rating Prediction")
    st.write("""
        This dashboard helps HR teams assess two key outcomes:
        - **Attrition**: Will the employee stay or leave?
        - **Performance**: How is the employee likely to perform?
    """)

# 👤 Attrition Prediction
elif page == 'Employee Attrition Predictor':
    st.title("👤 Employee Attrition Prediction")
    input_dict = collect_inputs(for_attrition=True)

    if st.button("Predict Attrition"):
        try:
            X_scaled = preprocess_input(input_dict, for_attrition=True)
            pred = attrition_model.predict(X_scaled)[0]
            prob = attrition_model.predict_proba(X_scaled)[0][1]

            st.subheader("Prediction Result")
            if pred == 1:
                st.warning("⚠️ This employee is likely to leave the company.")
            else:
                st.success("✅ This employee is likely to stay.")

            st.write(f"Confidence: {prob:.2%}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# 📈 Performance Rating
elif page == 'Performance Rating Predictor':
    st.title("📈 Performance Rating Prediction")
    input_dict = collect_inputs(for_attrition=False)

    if st.button("Predict Performance Rating"):
        try:
            X_scaled = preprocess_input(input_dict, for_attrition=False)
            rating = performance_model.predict(X_scaled)[0]
            probas = performance_model.predict_proba(X_scaled)[0]

            st.subheader("Predicted Rating")
            st.success(f"🌟 Performance Rating: {rating}")
            st.write(f"Confidence: {np.max(probas):.2%}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")