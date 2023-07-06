import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# BIP Project 6 - Karell, Antonio, Hanna
# adapted from from https://www.youtube.com/watch?v=ZE2DANLfBIs&ab_channel=NeuralNine

# Install required packages
!pip install huggingface
!pip install huggingface_hub
!pip install transformers
!pip install datasets
# Load dataset
from datasets import load_dataset
data = load_dataset("liar")

# Simplify the labeling to binary
labelarray = []
for i in range(len(data["test"]["label"])):
    if data["test"]["label"][i] == 3:
        labelarray.append(0)
    else:
        labelarray.append(1)

X, y = data['test']['statement'], labelarray
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Create TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Create and train LinearSVC classifier
classifier = LinearSVC()
classifier.fit(X_train_vectorized, y_train)

# Check accuracy
accuracy = classifier.score(X_test_vectorized, y_test)
print("Accuracy:", accuracy)

# Predict on new text
text = "Some new text to predict"
vectorized_text = vectorizer.transform([text])
prediction = classifier.predict(vectorized_text)
print("Prediction:", prediction)
