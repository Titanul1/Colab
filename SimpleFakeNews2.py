import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import subprocess
from datasets import load_dataset

# Install required packages
subprocess.run(["pip", "install", "huggingface"])
subprocess.run(["pip", "install", "huggingface_hub"])
subprocess.run(["pip", "install", "transformers"])
subprocess.run(["pip", "install", "datasets"])

# Load dataset
data = load_dataset("liar")

# Simplify the labeling to binary
label_array = []
for i in range(len(data["test"]["label"])):
    if data["test"]["label"][i] == 3:
        label_array.append(0)
    else:
        label_array.append(1)

X, y = data['test']['statement'], label_array

# Shuffle the indices of your data
indices = np.random.permutation(len(X))

# Define the ratio for the train-test split
train_ratio = 0.8  # 80% for training, 20% for testing

# Compute the split index
split_index = int(len(X) * train_ratio)

# Split the data and labels into training and testing sets
X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]

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
