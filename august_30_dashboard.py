# Import the neccesary modules
import streamlit as st
import pandas as pd
# from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

! pip install scikit-learn

# Load and prepare
# iris = load_iris()
df=pd.read_csv('iris.csv')


# df = pd.DataFrame(data = iris.data, columns= iris.feature_names)
# df["species"] = iris.target
# df['species'] = df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

# Sidebar from user input
st.sidebar.header("Input Features")

def user_input_features():
    sepal_length = st.sidebar.slider('sepal_length', float(df['sepal_length'].min()), float(df['sepal_length'].max()), float(df['sepal_length'].mean()))
    sepal_width = st.sidebar.slider('sepal_width', float(df['sepal_width'].min()), float(df['sepal_width'].max()), float(df['sepal_width'].mean()))
    petal_length = st.sidebar.slider('petal_length', float(df['petal_length'].min()), float(df['petal_length'].max()), float(df['petal_length'].mean()))
    petal_width = st.sidebar.slider('petal_width', float(df['petal_width'].min()), float(df['petal_width'].max()), float(df['petal_width'].mean()))

    data = {'sepal_length' : sepal_length,
            'sepal_width' : sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index = [0])
    return features

input_df = user_input_features()

# Main Panel 
st.write("# Iris Flower Prediction")

# Combine the input features with the entire Dataset
iris_raw = df.drop(columns=['species'])
iris_raw = pd.concat([input_df, iris_raw], axis=0)

# Standadize the input features
scaler = StandardScaler()
iris_raw_scaled = scaler.fit_transform(iris_raw)
input_scaled = iris_raw_scaled[:1] #select only the user input

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(iris_raw_scaled[1:], df['species'])

# Predict 
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

st.subheader("Prediction")
# st.write(iris.target_names[prediction])

st.subheader("Prediction Probability")
st.write(prediction_proba)