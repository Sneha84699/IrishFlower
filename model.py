import streamlit as st
import numpy as np
import pickle
import os

# Load the model if the file exists
if os.path.exists("mymodel.pkl"):
    try:
        clf = pickle.load(open("mymodel.pkl", "rb"))
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        clf = None  # Set clf to None if there's an issue loading the model
else:
    st.error("Model file 'mymodel.pkl' not found. Please check the file path.")
    clf = None  # Set clf to None if the model file isn't found

def predict(data):
    if clf is not None:
        try:
            return clf.predict(data)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            return None
    else:
        st.error("Model is not loaded. Prediction cannot be made.")
        return None

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
    if result is not None:
        st.text(f"Predicted Species: {result[0]}")
