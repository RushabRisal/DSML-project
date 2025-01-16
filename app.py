import streamlit as st
import pandas as pd

st.text('Visualization of the data')
dataframe= pd.read_csv('./data/tarkari.csv')
st.dataframe(dataframe.head(100))
# st.line_chart(data = dataframe.head(),x=dataframe['Average'])
description= dataframe.describe()
st.write(description)
