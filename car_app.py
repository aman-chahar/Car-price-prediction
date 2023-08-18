#!/usr/bin/env python
# coding: utf-8

# Problem Statement:
# 
# 1. We are provided with a second-hand  car data set in that we have details like their name, brand, year of purchasing, current selling price by their owner,types of owner, km_driven that's how much they have travelled till now also the fuel type of car and seller type.
# 
# 2. We are required to build model for the cars with the available independent variables.
# 
# 3. We have to deploy the model on streamlit.

# In[3]:


#importing the libraries
import pandas as pd   # data preprocessing
import numpy as np    # mathematical computation
import pickle
from sklearn import *
import streamlit as st
import sys

# Load the model and dataset
model = pickle.load(open('la_model.pkl','rb'))
df = pickle.load(open('data.pkl','rb'))

st.title('Car Price Prediction')
st.header('Fill the details to predict car Price')




brand=st.selectbox('brand',df['brand'].unique())
year=st.selectbox('year is',[2017, 2012, 2015, 2014, 2013, 2018, 2016])
km_driven=st.selectbox('km_driven',df['km_driven'].unique())
fuel=st.selectbox('fuel',['Petrol','Diesel','CNG','LPG','Electric'])
seller_type=st.selectbox('seller_type',df['seller_type'].unique())
transmission=st.selectbox('transmission',df['transmission'].unique())
owner=st.selectbox('owner',df['owner'].unique())


if st.button('Predict Car Price'):
        

        pred = model.predict([[brand,year,km_driven,fuel,seller_type,transmission,owner]])
        output = round(pred[0],2)
        if pred < 0: # handeling negative outputs.
            st.error('The input values must be irrelevant, try again by giving relevent information.')
      
        write = str(pred) 
        st.success(write)







