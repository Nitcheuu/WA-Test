import pandas as pd
import numpy as np
import pickle
import streamlit as st
import os
import sklearn


print(sklearn.__version__)
def get_profil():

    Gender=st.sidebar.selectbox("Genre",("Male", "Female"))

    Married=st.sidebar.selectbox("Mariage",("Yes", "No"))

    Dependents=st.sidebar.selectbox("Nombre d'enfants", ("0","1","2","3+"))

    Education=st.sidebar.selectbox("Education",("Graduate", "Not Graduate"))

    Self_Employed=st.sidebar.selectbox("Salarié ou non", ("Yes", "No"))

    ApplicantIncome=st.sidebar.slider("Salaire",0,10000,5000)

    CoapplicantIncome = st.sidebar.slider("Salaire du conjoint", 0, 10000, 5000)

    LoanAmount=st.sidebar.slider("Montant du crédit en milliers de $", 5.0, 700.0, 350.0)

    Loan_Amount_Term=st.sidebar.selectbox("Durée du crédit en mois",(12.0,36.0,60.0,84.0,120.0,180.0,240.0,300.0,360.0))

    Credit_History=st.sidebar.selectbox("Credit_Hisotry", (1.0,0.0))

    Property_Area=st.sidebar.selectbox("Secteur de la propriété",("Urban", "Rural","Semiurban"))

    data={
        'Gender' : Gender,
        'Married' : Married,
        'Dependents' : Dependents,
        'Education' : Education,
        'Self_Employed' : Self_Employed,
        'ApplicantIncome' : ApplicantIncome,
        'CoapplicantIncome' : CoapplicantIncome,
        'LoanAmount' : LoanAmount,
        'Loan_Amount_Term' : Loan_Amount_Term,
        'Credit_History' : Credit_History,
        'Property_Area' : Property_Area
    }

    profil_client = pd.DataFrame(data, index=[0])

    return profil_client


st.title("Test")

st.sidebar.header("Votre profil")

input_df = get_profil()

df = pd.read_csv("test.csv")
credit_input = df.drop(columns=['Loan_ID', 'Loan_Status'])
donnee_entree = pd.concat([input_df, credit_input], axis=0)
var_cat = ['Gender', 'Married', 'Dependents', 'Education','Self_Employed', 'Credit_History', 'Property_Area']
var_num = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term']
for col in var_cat:
    dummy = pd.get_dummies(donnee_entree[col], drop_first=True)
    donnee_entree = pd.concat([dummy, donnee_entree], axis=1)
    del donnee_entree[col]

donnee_entree = donnee_entree[:1]

st.subheader("Les caractéristiques : ")
st.write(donnee_entree)

path = "prevision_credit.pkl"
with open(path, 'rb') as file:
    model = pickle.load(file)

prevision = model.predict(donnee_entree)

st.subheader("Résultat :")
if prevision == 0:
    st.write("Crédit accordé !")
else:
    st.write("Crédit refusé !")