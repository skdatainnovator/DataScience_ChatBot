import streamlit as st
import pickle
import pandas as pd
import numpy as np
import string
import re
import json
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.linear_model import SGDClassifier
from collections import Counter
import nlpaug.augmenter.word as naw
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score
import gzip


def load_model():
    model2 = open('DS_chatbot.pkl','rb')

    forest = pickle.load(model2)
    return forest

model = load_model()

##Code for data preparation
# lemmatizer = WordNetLemmatizer()
le = LabelEncoder()
final_df = pd.read_csv("augmented1.csv")
tf = TfidfVectorizer(ngram_range=(1, 3),min_df=0,stop_words='english')
X = final_df['question']
y = final_df['answer']
y = le.fit_transform(y)
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)
X = tf.fit_transform(X)
#X_test_tf = tf.transform(X_test)


# def clean_data(text):
#     text=text.lower() #lower the text
#     text = re.sub(r'[^\w\s]', '', text) #remove irrelevant characters    
#     text = text.split() #convert sentence to tokens
#     text = [lemmatizer.lemmatize(word) for word in text] #lemmatization
#     text = " ".join(text) #converting tokens to sentence
#     return text

#Chat Page

def chat_page():
    st.title("Data Science FAQ Chatbot")
    st.write("""### Please Enter your Data Science related Query""")
    chat_que = st.text_input('Enter your Query','hello')
    #chat_que = clean_data(chat_que)
    chat_que = tf.transform([chat_que])
    if np.amax(model.predict_proba(chat_que))>0.2:
        chat_ans = le.inverse_transform(model.predict(chat_que))[0]
        st.subheader(f"Answer : {chat_ans}")
    else:
        st.subheader(f"Not So Sure About This!! Below Answer Might Help")
        chat_ans = le.inverse_transform(model.predict(chat_que))[0]
        st.subheader(f"Answer : {chat_ans}")
        
