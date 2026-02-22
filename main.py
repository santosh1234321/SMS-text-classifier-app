import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LENGTH = 50

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('sms_model.keras')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

st.title("🛡️ SMS Spam Detector")
model, tokenizer = load_assets()

user_input = st.text_area("Enter SMS message:")

if st.button("Analyze"):
    if user_input:
        # 1. Transform text using the SAVED tokenizer
        sequences = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post')
        
        # 2. Predict
        prob = model.predict(padded, verbose=0)[0][0]
        
        # 3. Display Result
        st.write(f"Raw Probability Score: `{prob:.4f}`")
        if prob >= 0.5:
            st.error(f"RESULT: SPAM (Confidence: {prob:.2%})")
        else:
            st.success(f"RESULT: HAM (Confidence: {1-prob:.2%})")
