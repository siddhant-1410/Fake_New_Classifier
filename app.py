import streamlit as st
# import tensorflow as tf
# import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re


import tensorflow as tf
import pickle

# Test loading the model
try:
    model = tf.keras.models.load_model('lstm.keras')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Test loading the tokenizer
try:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")

# Preprocessing function
def preprocess_text(text):
    
    text = re.sub(r'[^a-zA-Z\s]', '', text)
   
    text = text.lower()
    
    text = ' '.join(text.split())
    return text

# Streamlit UI
st.title("Fake News Detection")

input_text = st.text_area("Enter news text to check if it's fake or real",height=400)

if st.button("Predict"):
    if input_text.strip() == "":
        st.write("Please enter some text.")
    else:
        # Preprocess the input text
        processed_text = preprocess_text(input_text)
        
       
        sequences = tokenizer.texts_to_sequences([processed_text])
        maxlen = 1000  
        
        
        padded = pad_sequences(sequences, maxlen=maxlen)
        print(len(padded))
        x = np.array(padded)
       
        prediction = (model.predict(x) >= 0.5).astype(int)
        
        # Interpret the prediction
        if prediction == 1:
            st.markdown('<p style="color:green;">The news is likely Real</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:red;">The news is likely Fake.</p>', unsafe_allow_html=True)
