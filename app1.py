import streamlit as st
import numpy as np
import pandas as pd
import pickle


my_dtr = pickle.load(open("weather_dtr.pkl","rb"))

def predict(data):
    return my_dtr.predict(data)

st.title("Create a model that will help me predict relative humidity at 3 pm")
st.markdown("This model predicts humidity at 3pm")


st.header("Independed Variables")


col1, col2 = st.columns(2)


with col1:
    ap_9am = st.slider("Air Pressure at 9am", 1.0, 1000.0, 0.2)
    at_9am = st.number_input("Air Temp at 9am", 0.0, 100.0, 0.1)
    awd_9am = st.slider("Avg Wind Direction at 9am", 10.0, 400.0, 0.1)
    aws_9am = st.number_input("Avg Wind Speed at 9am", 0.0, 100.0, 0.2)

with col2:
    mwd_9am = st.slider("Max Wind Direction at 9am", 25.0, 400.0, 0.2)
    mws_9am = st.number_input("Max Wind Speed at 9am", 0.0, 50.0, 0.2)
    ra_9am = st.slider("Rain Accumulation at 9am", 1.0, 10.0, 0.1)
    rd_9am = st.slider("Rain Duration at 9am", 1.0, 7000.0, 0.1)
    rh_9am = st.number_input("Relative Humidity at 9am", 0.0, 100.0, 0.2)


# Prediction button
if st.button("Here is Humidity at 3pm"):
    result = my_dtr.predict(np.array([[ap_9am,at_9am,awd_9am,aws_9am,mwd_9am,mws_9am,ra_9am,rd_9am,rh_9am]]))
    st.text(result[0])
   
