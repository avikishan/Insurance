import streamlit as st
import numpy as np
import pandas as pd
import pickle
model = pickle.load(open('model.pkl','rb'))
transformer = pickle.load(open("transformer.pkl","rb"))
target_encoder = pickle.load(open("target_encoder.pkl",'rb'))


st.title("Insurance Premium Prediction")

Gender = st.selectbox("Please Select your Gender",('male','female'))
age = st.selectbox("Enter your Age",tuple([i for i in range(0,100)]))
bmi = st.text_input("Enter your BMI",20)
bmi = float(bmi)

children = st.selectbox("Please Select Number of Children",(0,1,2,3,4,5,6))
children=int(children)

smoker = st.selectbox("Please Select Smoker Category",('Yes','No'))

region =st.selectbox("Please Select your Region",('southwest','southeast','northwest','northeast'))

l = {
    'age':age,
    'sex':Gender,
    'bmi':bmi,
    'children':children,
    'smoker':smoker,
    'region':region
}
df=pd.DataFrame(l,index=[0])

df['region']=target_encoder.transform(df['region'])
df['sex']=df['sex'].map({'male':1,'female':0})
df['smoker']=df['smoker'].map({'Yes':1,'No':0})


df = transformer.transform(df)

y_pred = model.predict(df)

if st.button("Show Results"):
    st.header(f"{round(y_pred[0],2)} INR")


