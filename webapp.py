import streamlit as st
import time
import numpy as np
import pandas as pd
import re
import nltk
import sklearn

from sklearn.feature_extraction.text import CountVectorizer
import string
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords')
from nltk.corpus import stopwords

# NLP MODEL

df = pd.read_csv('https://docs.google.com/spreadsheets/d/1h20jiCScTTrfSxhOwM5vJrP9vXamUu3cSQlmtKEY6VI/edit?usp=sharing', encoding='latin1')
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df.rename(columns={'v1': 'labels', 'v2': 'message'}, inplace=True)
df.drop_duplicates(inplace=True)

df['labels'] = df['labels'].map({'ham':0, 'spam':1})

# remove punctuation and clean

def clean_data(message):
    message_without_punctuation = [character for character in message if character not in string.punctuation]
    message_without_punctuation = "".join(message_without_punctuation)

    seperator = " "
    return seperator.join([word for word in message_without_punctuation.split() if word.lower() not in stopwords.words('english')])

df['message'] = df['message'].apply(clean_data)

x = df['message']
y = df['labels']

cv = CountVectorizer()

x = cv.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

model = MultinomialNB().fit(x_train, y_train)

predictions = model.predict(x_test)


def predict(text):
    labels = ['Not a Spam', "a Spam"]
    x = cv.transform(text).toarray()
    p = model.predict(x)
    s = [str(i) for i in p]
    v = int("".join(s))

    return str("This message appears to be " + labels[v])


st.title("Spam Classifier")
st.image('image.jpg')

user_input = st.text_input("Write your message ")
submit = st.button('Predict')

if submit:
    answer = predict([user_input])
    st.text(answer)

# rerun.
st.button("Re-run")
