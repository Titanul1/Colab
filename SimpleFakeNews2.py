{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count":
      "metadata" 
        "id": "zNUR-hw4MHo1"
      
      "outputs": [],
      "source": [
        "# BIP Project 6 - Karell, Antonio, Hanna\n",
        "# adapted from from https://www.youtube.com/watch?v=ZE2DANLfBIs&ab_channel=NeuralNine\n",
        "\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "!pip install huggingface\n",
        "!pip install huggingface_hub\n",
        "!pip install transformers\n",
        "!pip install datasets\n",
        "#change text into a vector of features (term frequency, inverse document frequency)\n",
        "#term frequency (how important/common is this word), IDF - how frequent is this word compared to other docs (how distinctive is it???)\n",
        "\n",
        "from sklearn.svm import LinearSVC #an algorithm for method of classifying, maximizing the distance between classifying samples (linear support vector)\n",
        "#other variants possible: K nearest neighbors, random forest... supposedly LinearSVC gave the best results"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from datasets import load_dataset\n",
        "\n",
        "data = load_dataset(\"liar\")\n",
        "#print(data[\"test\"][\"label\"])\n",
        "#using the LIAR dataset - https://paperswithcode.com/dataset/liar\n",
        "#data['fake'] = data[\"test\"][\"label\"]  #using this model, it ended up being only 19% accuracy... perhaps because of having 4 gradations of truth?\n",
        "#data['text'] = data[\"test\"][\"statement\"]\n",
        "#data = data.drop(\"label\", axis=1) #relabeling fake to be a numerical value rather than real/fake\n",
        "\n",
        "\n",
        "### So this is simplifying the labeling to be just binary.\n",
        "print(data[\"test\"][\"label\"])\n",
        "labelarray = []\n",
        "\n",
        "for i in range(len(data[\"test\"][\"label\"])):\n",
        " if data[\"test\"][\"label\"][i] == 3:\n",
        "    labelarray.append(0)\n",
        " else:\n",
        "    labelarray.append(1)\n",
        "print(labelarray)\n",
        "\n",
        "#print(data[\"test\"][\"label\"])\n",
        "#data['fake'] = data[\"test\"][\"label\"].apply(lambda x: 0 if x == \"3\" else 1) #only 3 is true... 2 is mostly true, 1 4 5 are false\n",
        "#print(data[\"fake\"])\n",
        "#print(data[\"test\"][\"label\"])\n"
      ],
      "metadata": {
        "id": "lhtnAZ19NBhR"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X, y = data['text'], data['fake'] #identifying text column as input and fake column as output to predict\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10) #reserves 80% of the data for training and 20% for checking the generated classifying rules afterwards\n",
        "\n",
        "\n",
        "vectorizer = TfidfVectorizer(stop_words=\"english\") #changing training data text into numerical features\n",
        "\n",
        "X_train_vectorized = vectorizer.fit_transform(X_train)\n",
        "#fit_transform method helps standardize it and transform it to be centered on the mean\n",
        "X_test_vectorized = vectorizer.transform(X_test)\n",
        "#no need to vectorize y variables because we already converted them into numerical values of 0 and 1\n",
        "\n",
        "classifier = LinearSVC() #in the future, we could add grid search or randomized search to improve results\n",
        "classifier.fit(X_train_vectorized, y_train) #now we run the classifier algorithm on these numericized values of X and Y\n",
        "classifier.score(X_test_vectorized, y_test) #now that the model is trained, we check it against the test sample, to see accuracy %\n",
        "\n",
        "#much worse performance for tweets, much worse for precisely labeling in gradations of 4... 19%"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zeBjK2wBQhhI",
        "outputId": "faf84000-42a9-4037-97e3-9ab34431a7a1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.1867704280155642"
            ]
          },
          "metadata": {},
          "execution_count": 37
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#now let's try this on another text that hasn't been classified yet, to predict if it's fake\n",
        "gdown.download('https://drive.google.com/uc?id=18bU9lBfh_hrQHUfwzQwkjOVyN3VIQIPs', \"unidentified.txt\")\n",
        "with open(\"unidentified.txt\", \"r\", encoding=\"utf-8\") as f:\n",
        "  text = f.read()\n",
        "vectorized_text = vectorizer.transform([text]) #have to vectorize it again, to get numerical values for the algorithm\n",
        "#text is in the form of a collection when we do .read(), so make sure to put []\n",
        "\n",
        "classifier.predict(vectorized_text)\n",
        "#documentation for linearSVC says that .predict method returns likely class value of output\n",
        "#so in this case, result array([1]) means that it classified it as fake news ... and later it turned out to be indeed fake"
      ],
      "metadata": {
        "id": "W1N3nEtJUg2Q"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#here is a potential modification for determining what % of a site is fake news (giving a rough idea of how reputable a site may be--more likely real news or more likely clickbait)\n",
        "# Step 1 - Scrape a desired website (probably could enhance this with an API, but for now, using .csv from other scrapers... APIFY Smart Article Extractor)\n",
        "\n",
        "\n",
        "gdown.download('https://drive.google.com/uc?id=1YJ1qWFmPpMQb1cBgCa5gJl0bm4ow4EGD', 'guardianarticles.csv')\n",
        "\n",
        "Xdata = pd.read_csv('guardianarticles.csv')  # we can replace this dataset with other text\n",
        "Xdata_vectorized = vectorizer.transform(Xdata)\n",
        "#no need to vectorize y variables because we already converted them into numerical values of 0 and 1\n",
        "\n",
        "print(Xdata_vectorized)  ## why is this showing just one row of data, when Xdata had 84 rows?\n",
        "\n",
        "#X_vectorized = vectorizer.transform(data)\n",
        "#print(X_vectorized)\n",
        "#Y_predicted = classifier.predict(X_vectorized)\n",
        "#for i in range(len(Y_predicted)):\n",
        "#  print(Y_predicted[i])\n",
        "\n",
        "\n",
        "#for i in range(len(X_vectorized)):\n",
        "# print(\"X=%s, Predicted=%s\" % (X_vectorized[i], Y_predicted[i]))"
      ],
      "metadata": {
        "id": "fQSOkTqIYzB9"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
