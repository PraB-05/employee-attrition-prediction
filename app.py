import streamlit as st
import pandas as pd
import numpy as np

import pickle
with open('final_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('📊 Employee Attrition Predictor')
st.write('Predict wheather an employee is likely to leave')

st.subheader('Employee Detail')

columns = ['Age', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked', 'WorkLifeBalance', 'YearsAtCompany', 'IncomePerYear', 'Satisfaction', 'Department_Research & Development', 'Department_Sales', 'EducationField_Life Sciences', 'EducationField_Marketing', 'EducationField_Medical', 'EducationField_Other', 'EducationField_Technical Degree']

input_info = {col : 0 for col in columns}

education_map = {
    'Below College':1,
    'College' : 2,
    'Bachelor' : 3,
    'Master' : 4,
    'Doctor' : 5
}
satisfaction_map = {
    'Low' : 1,
    'Medium' : 2,
    'High' : 3,
    'Very high' : 4
}
work_life_map = {
    'Bad' : 1,
    'Good' : 2,
    'Better' : 3,
    'Best' : 4
}
marital_status_map = {
    'Single' : 0,
    'Married' : 1,
    'Divorced' : 2
}

age = st.number_input('Age', min_value=18 , max_value=70 , value=30)
distance = st.number_input('Distance' , min_value=1 , max_value=50 , value=5)
education_level = st.selectbox('Education level' , list(education_map.keys()) , index=2 )

education = education_map[education_level]
Environment_level = st.selectbox('Environment satisfaction' , list(satisfaction_map.keys()) , index=1)
Environment_satisfaction = satisfaction_map[Environment_level]

job_satisfaction_level = st.selectbox('Job satisfaction' , list(satisfaction_map.keys()) , index = 1)
job_satisfaction = satisfaction_map[job_satisfaction_level]

work_life_level = st.selectbox('Work life balance' , list(work_life_map.keys()) , index = 1)
work_life_balance = work_life_map[work_life_level]

marital_status_level = st.selectbox('Marital Status' , list(marital_status_map.keys()) , index=0)
marital_status = marital_status_map[marital_status_level]

monthly_income = st.number_input('Monthly Income' , min_value=1000 , max_value=20000 , value=2000)

NumberOfcompanies = st.number_input('Number of Companies Worked' , min_value=0 , max_value=10 , value=2)

yearsAtcompany = st.number_input('Years at Company' , min_value=0 , max_value= 40 , value= 5)

Department = st.selectbox('Department' , ['Human Resources', 'Research & Development', 'Sales'])
education_field = st.selectbox('Education Field' , ['Human Resources', 'Life Sciences', 'Marketing', 'Medical', 'Other', 'Technical Degree'])

# Engineered Features

income_per_year = monthly_income/age
satisfaction_score = (Environment_satisfaction + job_satisfaction +work_life_balance)/3

# numeric/base field

input_info['Age'] = age
input_info['DistanceFromHome'] = distance
input_info['Education'] = education
input_info['EnvironmentSatisfaction'] = Environment_satisfaction
input_info['JobSatisfaction'] = job_satisfaction
input_info['WorkLifeBalance'] = work_life_balance
input_info['MaritalStatus'] = marital_status
input_info['MonthlyIncome'] = monthly_income
input_info['NumCompaniesWorked'] = NumberOfcompanies
input_info['YearsAtCompany'] = yearsAtcompany
input_info['IncomePerYear'] = income_per_year
input_info['Satisfaction'] = satisfaction_score


# One hot encoding 
# distance
if Department == 'Research & Development':
    input_info['Department_Research & Development'] = 1 
else: 0

if Department == 'Sales':
    input_info['Department_Sales'] = 1
else: 0

# Education_field

input_info['EducationField_Life Sciences'] = 1 if education_field == 'Life Sciences'  else 0
input_info['EducationField_Marketing'] = 1 if education_field == 'Marketing'  else 0
input_info['EducationField_Medical'] = 1 if education_field == 'Medical'  else 0
input_info['EducationField_Other'] = 1 if education_field == 'Other'  else 0
input_info['EducationField_Technical Degree'] = 1 if education_field == 'Technical Degree'  else 0

input_df = pd.DataFrame([input_info] , columns=columns)

st.subheader('Prediction')
threshold = st.slider('Decision Threshold', min_value=0.10 , max_value=0.90, value=0.50 , step=0.1)


if st.button('Predict Attrition'):

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prob > threshold:
        st.error(f'⚠️ High Risk : Employee likely to leave')
        st.write(f'Attrition Probability: {prob:.2f}')
    else:
        st.success(f'✅ Low Risk : Employee likely to stay')
        st.write(f'Attrition Probability: {prob:.2f}')
