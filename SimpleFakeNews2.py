import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import subprocess

subprocess.run(["pip", "install", "scikit-learn"])

# Import the required module
from sklearn.model_selection import train_test_split

# Install packages using pip
subprocess.run(["pip", "install", "huggingface"])
subprocess.run(["pip", "install", "huggingface_hub"])
subprocess.run(["pip", "install", "transformers"])
subprocess.run(["pip", "install", "datasets"])

# BIP Project 6 - Karell, Antonio, Hanna
# adapted from from https://www.youtube.com/watch?v=ZE2DANLfBIs&ab_channel=NeuralNine
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
