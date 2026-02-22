# SMS Spam Detection Web Application

An Natural Language Processing (NLP) application designed to classify SMS messages as "ham" (legitimate) or "spam" (junk/fraudulent) with high precision. This project has been engineered into a full-stack AI web application, bridging the gap between model development and real-world deployment.

## 🚀 The Deployment Challenge
One of the key technical hurdles addressed in this project was ensuring **feature consistency** between training and deployment. By replacing non-deterministic hashing (like `one_hot`) with a saved **Tokenizer object**, the model maintains 100% accuracy across different Python environments.



## 🛠️ Technical Stack
* **Deep Learning:** TensorFlow / Keras
* **NLP Techniques:** Tokenization, Sequence Padding, Word Embeddings
* **Web Framework:** Streamlit
* **Data Management:** Pandas, NumPy, Pickle

## 🏗️ Architecture & Model
The model uses a **Global Average Pooling** architecture, which is highly effective for short-text classification like SMS messages.

* **Embedding Layer:** Maps words to a 64-dimensional vector space.
* **GlobalAveragePooling1D:** Extracts the "essence" of the message regardless of word position. 
* **Dense Layers:** ReLU-activated hidden layers for pattern recognition.
* **Output:** Sigmoid activation for binary probability (0 to 1). 

