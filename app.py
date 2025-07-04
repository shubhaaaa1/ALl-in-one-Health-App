import streamlit as st
import pickle
import numpy as np

# Load the trained models
calorie_model = pickle.load(open('calorie_model.pkl', 'rb'))
diabetes_model = pickle.load(open('diabetes_model.pkl', 'rb'))
heart_model = pickle.load(open('heart_model.pkl', 'rb'))
insurance_model = pickle.load(open('MedicalInsuranceCost.pkl', 'rb'))
cancer_model = pickle.load(open('breast_cancer.pkl', 'rb'))

# Streamlit App Configuration
st.set_page_config(page_title="All-in-One Health Prediction App", layout="wide")

st.title("💻 All-in-One Health Prediction App")
st.sidebar.title("Choose Prediction Type")

# Sidebar Navigation
app_mode = st.sidebar.radio("Select Prediction", 
    ['Home', 
     'Calorie Burn Prediction', 
     'Diabetes Prediction', 
     'Heart Disease Prediction', 
     'BMI Calculator', 
     'Medical Insurance Cost Prediction', 
     'Breast Cancer Classification'])

# Home Page
if app_mode == 'Home':
    st.write("""
    ## Welcome to the All-in-One Health Prediction App! 🚀
    This app provides:
    - 🔥 Calorie Burn Prediction
    - 💉 Diabetes Risk Prediction
    - ❤️ Heart Disease Prediction
    - ⚖️ BMI Calculator
    - 💰 Medical Insurance Cost Prediction
    - 🩺 Breast Cancer Classification
    """)

# Calorie Burn Prediction
elif app_mode == 'Calorie Burn Prediction':
    st.subheader("🔥 Calorie Burn Prediction")
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.number_input('Age', min_value=10, max_value=100, step=1)
    height = st.number_input('Height (cm)', min_value=100.0, max_value=250.0, step=1.0)
    weight = st.number_input('Weight (kg)', min_value=30.0, max_value=200.0, step=1.0)
    duration = st.number_input('Duration of Exercise (minutes)', min_value=1, max_value=300, step=1)
    heart_rate = st.number_input('Heart Rate', min_value=60, max_value=200, step=1)
    body_temp = st.number_input('Body Temperature (°C)', min_value=36.0, max_value=42.0, step=0.1)

    gender_val = 1 if gender == 'Male' else 0  # Label Encoding

    if st.button('Predict Calories Burnt'):
        input_data = np.array([[gender_val, age, height, weight, duration, heart_rate, body_temp]])
        result = calorie_model.predict(input_data)
        st.success(f'Calories Burnt: {result[0]:.2f} kcal')

# Diabetes Prediction
elif app_mode == 'Diabetes Prediction':
    st.subheader("💉 Diabetes Risk Prediction")
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, step=1)
    glucose = st.number_input('Glucose Level', min_value=50, max_value=300, step=1)
    blood_pressure = st.number_input('Blood Pressure', min_value=30, max_value=200, step=1)
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, step=1)
    insulin = st.number_input('Insulin Level', min_value=0, max_value=900, step=1)
    bmi = st.number_input('BMI', min_value=10.0, max_value=70.0, step=0.1)
    diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, step=0.01)
    age = st.number_input('Age', min_value=10, max_value=100, step=1)

    if st.button('Predict Diabetes'):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
        result = diabetes_model.predict(input_data)
        if result[0] == 1:
            st.error('High Risk of Diabetes')
        else:
            st.success('Low Risk of Diabetes')

# Heart Disease Prediction
elif app_mode == 'Heart Disease Prediction':
    st.subheader("❤️ Heart Disease Prediction")
    age = st.number_input('Age', min_value=10, max_value=100, step=1)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    cp = st.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3])
    trestbps = st.number_input('Resting Blood Pressure', min_value=50, max_value=250, step=1)
    chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, step=1)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No'])
    restecg = st.selectbox('Resting ECG Results (0-2)', [0, 1, 2])
    thalach = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=250, step=1)
    exang = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
    oldpeak = st.number_input('ST Depression', min_value=0.0, max_value=6.0, step=0.1)
    slope = st.selectbox('Slope of Peak Exercise ST Segment', [0, 1, 2])
    ca = st.selectbox('Number of Major Vessels (0-3)', [0, 1, 2, 3])
    thal = st.selectbox('Thalassemia (0-3)', [0, 1, 2, 3])

    sex_val = 1 if sex == 'Male' else 0  # Label Encoding
    fbs_val = 1 if fbs == 'Yes' else 0
    exang_val = 1 if exang == 'Yes' else 0

    if st.button('Predict Heart Disease'):
        input_data = np.array([[age, sex_val, cp, trestbps, chol, fbs_val, restecg, thalach, exang_val, oldpeak, slope, ca, thal]])
        result = heart_model.predict(input_data)
        if result[0] == 1:
            st.error('High Risk of Heart Disease')
        else:
            st.success('Low Risk of Heart Disease')

# BMI Calculator
elif app_mode == 'BMI Calculator':
    st.subheader("⚖️ BMI Calculator")
    height = st.number_input('Height (cm)', min_value=100.0, max_value=250.0, step=1.0)
    weight = st.number_input('Weight (kg)', min_value=30.0, max_value=200.0, step=1.0)

    if st.button('Calculate BMI'):
        height_m = height / 100
        bmi = weight / (height_m ** 2)
        st.info(f'Your BMI is {bmi:.2f}')

        if bmi < 18.5:
            st.warning('Underweight')
        elif 18.5 <= bmi < 24.9:
            st.success('Normal weight')
        elif 25 <= bmi < 29.9:
            st.warning('Overweight')
        else:
            st.error('Obese')

# Medical Insurance Cost Prediction
elif app_mode == 'Medical Insurance Cost Prediction':
    st.subheader("💰 Medical Insurance Cost Prediction")
    age = st.number_input('Age', min_value=10, max_value=100, step=1)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, step=0.1)
    children = st.number_input('Number of Children', min_value=0, max_value=10, step=1)
    smoker = st.selectbox('Smoker', ['Yes', 'No'])
    region = st.selectbox('Region', ['northeast', 'northwest', 'southeast', 'southwest'])

    sex_val = 1 if sex == 'Male' else 0  # Label Encoding
    smoker_val = 1 if smoker == 'Yes' else 0

    # Make sure this region mapping matches your label encoder!
    region_mapping = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
    region_val = region_mapping[region]

    if st.button('Predict Insurance Cost'):
        input_data = np.array([[age, sex_val, bmi, children, smoker_val, region_val]])
        result = insurance_model.predict(input_data)
        st.success(f'Estimated Insurance Cost: ${result[0]:.2f}')

# Breast Cancer Classification
elif app_mode == 'Breast Cancer Classification':
    st.subheader("🩺 Breast Cancer Classification")
    st.write("Enter the following features:")

    features = []
    feature_names = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
    
    for feature in feature_names:
        value = st.number_input(f'{feature}', min_value=0.0)
        features.append(value)

    if st.button('Predict Breast Cancer'):
        input_data = np.array([features])
        result = cancer_model.predict(input_data)
        if result[0] == 1:
            st.error('Malignant (High Risk)')
        else:
            st.success('Benign (Low Risk)')
