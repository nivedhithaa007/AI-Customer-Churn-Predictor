import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert Yes/No churn into 1/0
data['Churn'] = data['Churn'].map({
    'Yes':1,
    'No':0
})

# Convert text columns into numbers
encoder = LabelEncoder()

columns = ['gender',
           'Partner',
           'Dependents',
           'PhoneService']

for col in columns:
    data[col] = encoder.fit_transform(data[col])

# Select features
X = data[['tenure',
          'MonthlyCharges',
          'gender',
          'Partner',
          'Dependents',
          'PhoneService']]

y = data['Churn']

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X,y)

st.title("AI Customer Churn Predictor")

st.subheader("Enter Customer Details")

tenure = st.number_input(
    "Subscription Months",
    min_value=0
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0
)

gender = st.selectbox(
    "Gender",
    ["Male","Female"]
)

partner = st.selectbox(
    "Partner",
    ["Yes","No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes","No"]
)

phone = st.selectbox(
    "Phone Service",
    ["Yes","No"]
)

# Convert input values
gender = 1 if gender=="Male" else 0
partner = 1 if partner=="Yes" else 0
dependents = 1 if dependents=="Yes" else 0
phone = 1 if phone=="Yes" else 0

if st.button("Predict"):

    prediction=model.predict(
        [[tenure,
          monthly,
          gender,
          partner,
          dependents,
          phone]]
    )

    if prediction[0]==1:
        st.error("Customer likely to leave ❌")

    else:
        st.success("Customer likely to stay ✅")