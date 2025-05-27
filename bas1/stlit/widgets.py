import streamlit as st
import pandas as pd

st.title("Streamlit Text Input")

name = st.text_input("Enter your name: ","'Your Name'")

age = st.slider("Select your Age: ",0,100,25)

options= ['python','java','c++','go']
choice = st.selectbox("Choose your Favourite language:", options)

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Boston', 'Chicago']
}
df = pd.DataFrame(data)
df.to_csv('sample_data.csv')
if name:
    st.write(f"Hello {name}")
if age:
    st.write(f"your Age is {age}")
st.write(f"you selected {choice}")

st.write(df)
st.line_chart(df)

## uploader
upload_file = st.file_uploader("Choose a csv file",'csv')

if upload_file is not None:
    df2 = pd.read_csv(upload_file)
    st.write(df2)