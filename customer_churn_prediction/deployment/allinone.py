import streamlit as st
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np

def set_bg_hack_url():   
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(https://www.ppt-backgrounds.net/thumbs/light-blue-presentation-templates-backgrounds.jpg), url(https://img.freepik.com/free-vector/isometric-people-working-with-technology_52683-19078.jpg?w=2000);
             background-repeat: no-repeat;
             background-size: 800px 800px, cover;
             background-position: center, center;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

set_bg_hack_url()

with open("prep_pipeline.pkl", "rb") as prep_file:
    preprocess = pickle.load(prep_file)

save_model = load_model('model_deep.h5')

columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges']

st.markdown("<h1 style='text-align: center; color: black;'> ➡️Telco Customer Churn Predictor⬅️ </h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: left; color: black;'> About Customers </h3>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
gen = col1.radio(
     "Gender",
     ('Male', 'Female'))

age = col2.slider('Age', 0, 130, 25)
if age>50:
    age = 0
else:
    age = 1

status = col3.radio(
     "Status",
     ('Partnered', 'Non-Partnered'))
if status=='Partnered':
    status = 1
else:
    status = 0

depend = col4.radio(
     "Dependents?",
     ('Yes', 'No'))
if depend=='Yes':
    depend = 1
else:
    depend = 0

st.markdown("<h3 style='text-align: left; color: black;'> Service </h3>", unsafe_allow_html=True)

col11, col21, col31 = st.columns([1, 1, 1])
phone = col11.radio(
     "Phone Services",
     ('Yes', 'No'))
multiple = col21.radio(
     "Multiple Lines",
     ('Yes', 'No', 'No phone service'))
internet = col31.radio(
     "Internet Services",
     ('DSL', 'Fiber optic', 'No'))

col12, col22, col32 = st.columns([1, 1, 1])
secure = col12.radio(
     "Online Security",
     ('Yes', 'No', 'No internet service'))
backup = col22.radio(
     "Online Backup",
     ('Yes', 'No', 'No internet service'))
protect = col32.radio(
     "Device Protection",
     ('Yes', 'No', 'No internet service'))

col13, col23, col33 = st.columns([1, 1, 1])
tech = col13.radio(
     "Tech Support",
     ('Yes', 'No', 'No internet service'))
tv = col23.radio(
     "TV Streaming",
     ('Yes', 'No', 'No internet service'))
movie = col33.radio(
     "Movie Streaming",
     ('Yes', 'No', 'No internet service'))

st.markdown("<h3 style='text-align: left; color: black;'> Account Information </h3>", unsafe_allow_html=True)

col14, col24, col34 = st.columns([1, 1, 1])
contrac = col14.radio(
     "Contract",
     ('Month-to-month', 'One year', 'Two year'))
billing = col24.radio(
     "Paperless Billing",
     ('Yes', 'No'))
if billing=='Yes':
    billing = 1
else:
    billing = 0
    
method = col34.selectbox("Payment Methods", 
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

col15, col25, col35 = st.columns([1, 1, 1])
monthly = col15.number_input("Monthly Charges")
total = col25.number_input("Total Charges")
tenur = col35.slider('Tenure', 0, 100, 0)

st.markdown("<h5 style='text-align: center; color: black;'> Click bellow to start predict </h5>", unsafe_allow_html=True)
col16, col27, col38 = st.columns([1.5, 1, 1])
if col27.button('Predict'):
    new_data = [gen, age, status, depend, tenur, phone,
            multiple, internet, secure, backup, protect, tech, 
            tv, movie, contrac, billing, method, monthly, total]
    new_datas = pd.DataFrame([new_data], columns=columns)
    new_data_prep = preprocess.transform(new_datas)
    pred = save_model.predict(new_data_prep)
    y_pred = np.where(pred > 0.5, 'Customer Leave', 'Customer Stay')[0][0]
else:
    y_pred = '-'
st.markdown(f"<h1 style='text-align: center; color: black;'> {y_pred} </h1>", unsafe_allow_html=True)

