# Loading and displaying data
import streamlit as st
import pandas as pd

# Title
st.title("Data Display in Streamlit")

# Load the data
path = '../Streamlit_dep/sample_dataset.csv'
df = pd.read_csv(path)

# Display the dataset
st.write("Dataset")
st.dataframe(df)

# Display the summary statistics of my dataset
st.write('Summary Statistics')
st.write(df.describe()) 