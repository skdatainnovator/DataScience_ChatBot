import streamlit as st

from chatbot import chat_page

page_number = st.sidebar.selectbox("DataScience or Data Analyst" , ("DataScience","Data Analyst"))

if page_number == "DataScience":
    chat_page()
else:
    pass