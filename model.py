import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained model
clf = pickle.load(open("model.pkl", "rb"))

def predict(data):
    return clf.predict(data)

# Streamlit interface for flower prediction
st.title("Flower Species Prediction using Machine Learning")
st.markdown("This model predicts the species of a flower based on its sepal and petal dimensions.")

st.header("Input Flower Measurements")
col1, col2 = st.columns(2)

with col1:
    st.text("Sepal Measurements")
    sepal_length = st.slider("Sepal Length (cm)", 0.0, 10.0, 0.5)
    sepal_width = st.slider("Sepal Width (cm)", 0.0, 10.0, 0.5)

with col2:
    st.text("Petal Measurements")
    petal_length = st.slider("Petal Length (cm)", 0.0, 10.0, 0.5)
    petal_width = st.slider("Petal Width (cm)", 0.0, 10.0, 0.5)

st.text('')

# Button to trigger prediction
if st.button("Predict Flower Species"):
    # Prepare the input data as a NumPy array
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    # Make prediction using the loaded model
    result = predict(input_data)
    # Display the prediction result
    st.text(f"Predicted Species: {result[0]}")

st.markdown("Developed by Rutuja & Sneha at NIELIT Daman")
