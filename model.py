import streamlit as st
import pandas as pd
import numpy as np
import pickle

# load the model
clf = pickle.load(open("model.pkl","rb"))

def predict(data):
    clf = pickle.load(open("model.pkl","rb"))
    return clf.predict(data)


st.title("Advertising spends prediction using Machine learning")
st.markdown("This model Identify total spends on advertising")

st.header("Advertising spends on various Media")
col1,col2 = st.columns(2)

with col1:
    st.text("TV")
    tv = st.slider("Adver. Spends on TV" , 1.0, 10000.0, 0.5)
    st.text("Radio")
    rd = st.slider("Adver. Spends on Radio" , 1.0, 10000.0, 0.5)
    st.text("Newspaper")
    newspaper = st.slider("Adver. Spends on Neswpaper" , 1.0,10000.0,0.5)
    
    
st.text('')
if st.button("Flower prediction "):
    result = clf.predict(np.array([[Sepal.Length, Sepal.Width, Petal.Length, Petal.Width]]))
    st.text(result[0])
st.markdown("Developed by rutuja & Sneha at NIELIT Daman")