import streamlit as st

st.title("Interactive Widgets")

# Text input
name = st.text_input("Enter Your name: ")

# Slider 
age = st.slider("Select your age: ", 18, 80, 30)


# Button 
if st.button("Submit"):
    st.write(f"Hello, {name}! You are {age} years old.")
