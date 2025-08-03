# Employee Attrition and Performance Rating Predictor

This project provides a machine learning-based solution to predict **employee attrition** and **performance rating** using historical HR data. It includes data preprocessing, model training, and a Streamlit web app for real-time predictions.

## 📁 Repository Structure

The repository contains the following files:

1. **`datapreprocessing.ipynb`**  
   Performs cleaning, encoding, and feature scaling on the HR dataset to prepare it for modeling.

2. **`attrition.ipynb`**  
   Trains and saves a machine learning model to predict whether an employee is likely to leave the organization (attrition prediction).

3. **`perfrating.ipynb`**  
   Trains and saves a machine learning model to predict the employee's performance rating.

4. **`app.py`**  
   A Streamlit-based web application that accepts employee input data and provides predictions for both attrition and performance rating.

## 🚀 Features

- Data preprocessing including label encoding and scaling
- Model training with Random Forest and XGBoost
- Streamlit interface for user-friendly predictions
- Modular code organization for flexibility and scalability

## ⚙️ Technologies Used

- Python
- Scikit-learn
- XGBoost
- Pandas, NumPy
- Streamlit
- Jupyter Notebooks

## 🚫 Excluded Files

The following files are excluded from the GitHub repo using `.gitignore`:

- `.pkl` files containing trained models and encoders
- Original dataset `.csv` files


# Employee Attrition & Performance Rating Prediction Dashboard

This is a Streamlit-based dashboard that allows HR teams to predict employee attrition (whether an employee is likely to leave) and performance rating using pre-trained machine learning models.

## Features
- **Employee Attrition Predictor:** Predicts the likelihood of an employee leaving the company.
- **Performance Rating Predictor:** Predicts the performance rating of an employee.
- **Interactive UI:** User-friendly input forms for all required employee features.
- **Model Confidence:** Displays prediction confidence for both attrition and performance.

## Project Structure
```
├── app.py                  # Main Streamlit dashboard application
├── pp.py                   # (Alternate/legacy) Streamlit app script
├── xgboost_model.pkl       # Pre-trained XGBoost model for attrition
├── rfc_model.pkl           # Pre-trained Random Forest model for performance
├── scaler.pkl              # Scaler used for feature normalization
├── label_encoders.pkl      # Label encoders for categorical features
├── Employee-Attrition - Employee-Attrition.csv  # Dataset (optional)
├── attrition.ipynb         # Data exploration / modeling notebook
├── perfrating.ipynb        # Performance rating modeling notebook
├── datapreprocessing.ipynb # Data preprocessing notebook
```


### Running the App
```
streamlit run app.py
```

The dashboard will open in your browser. Use the sidebar to navigate between Home, Attrition Predictor, and Performance Rating Predictor.

## Usage
1. Select the desired prediction tool from the sidebar.
2. Enter employee information in the form.
3. Click the prediction button to see results and model confidence.



