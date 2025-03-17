import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb

# load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

#load the pre-trained model with ReLU activation
model = load_model('simple_Rnn_imdb.h5')

## step-2 helper functions
# function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

# step-3 create our prediction function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment,prediction[0][0]


## streamlit app
import streamlit as st
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')


# user input 
user_input = st.text_input('Movie Review')

if st.button('Classify'):

    snetiment , score = predict_sentiment(user_input)
    
    # Display te prediction
    st.write(f'Sentiment : {snetiment}')
    st.write(f'Prediction score : {score}')
else:
    st.write('Please enter a movie review.')