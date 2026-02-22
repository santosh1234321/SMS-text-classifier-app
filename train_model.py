import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# 1. Download Data
def download_data():
    url_train = "https://cdn.freecodecamp.org/project-data/sms/train-data.tsv"
    url_test = "https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv"
    with open("train-data.tsv", "wb") as f:
        f.write(requests.get(url_train).content)
    with open("valid-data.tsv", "wb") as f:
        f.write(requests.get(url_test).content)

download_data()

# 2. Setup
VOCAB_SIZE = 10000
MAX_LENGTH = 50

train_df = pd.read_csv("train-data.tsv", sep='\t', names=['class', 'message'])
test_df = pd.read_csv("valid-data.tsv", sep='\t', names=['class', 'message'])

# 3. Use Tokenizer instead of one_hot (This is the key fix)
tokenizer = Tokenizer(num_words=VOCAB_SIZE, char_level=False)
tokenizer.fit_on_texts(train_df["message"])

# Save the tokenizer so the web app can use the EXACT same word-to-number mapping
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def prepare_data(df):
    sequences = tokenizer.texts_to_sequences(df["message"])
    padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post')
    labels = np.array([0 if x=="ham" else 1 for x in df['class']])
    return padded, labels

train_X, train_y = prepare_data(train_df)
test_X, test_y = prepare_data(test_df)

# 4. Model Architecture (Matches your Colab exactly)
model = Sequential([
    Embedding(VOCAB_SIZE, 64, input_length=MAX_LENGTH),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=20, verbose=1)

# 5. Save model
model.save('sms_model.keras')
print("Model and Tokenizer saved successfully!")
