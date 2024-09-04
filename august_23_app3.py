# Visualizing data in streamlit
# Import the libraries needed
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Title
st.title("Data Visualization with Streamlit")

# Load the data
path = './sample_dataset.csv'
df = pd.read_csv(path)

# Create a histogram using seaborn
fig, ax = plt.subplots()
sns.histplot(df['age'], bins=10, kde=True, ax = ax)
st.pyplot(fig)