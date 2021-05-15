import streamlit as st
from PIL import Image
import classify

st.title("Food-101 Classification.")
uploaded_file = st.file_uploader("Choose a Food image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    labels = classify.classify(uploaded_file)
    for i in labels:
        st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])