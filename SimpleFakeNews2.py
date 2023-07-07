import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC 
import gdown
import requests
import csv
import streamlit as st
import logging
from bs4 import BeautifulSoup


st.write("""
#Ceva pe acolo Boss
Hopa *CF GEIGELE?*""")


# Define constants
FAKE_REAL_NEWS_CSV_URL = 'https://drive.google.com/uc?id=14HFSVmD84uQai5IXDGBGHHvJ9SbEdPsA'
UNIDENTIFIED_TEXT_URL = 'https://drive.google.com/uc?id=18bU9lBfh_hrQHUfwzQwkjOVyN3VIQIPs'
GUARDIAN_ARTICLES_XLSX_URL = 'https://drive.google.com/uc?id=1GB1DG1Xxrxmw-ofRCQYZddYZahxr965k'
SUN_ARTICLES_XLSX_URL = 'https://drive.google.com/uc?id=1GKF8eNmxrLxB-Uba0YJ2yItMJTx4B_VJ'
INFOWARS_ARTICLES_XLSX_URL = 'https://drive.google.com/uc?id=1yiBXqnnem5uQjlp7NW5I0N6RUkVeS_Ux'
GUARDIAN_URL = 'https://www.theguardian.com/international'
ARTICLES_CSV_FILE = 'articles.csv'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url, output_file):
    """Download a file from a given URL."""
    try:
        gdown.download(url, output_file)
        logger.info(f'Successfully downloaded file: {output_file}')
    except Exception as e:
        logger.error(f'Error downloading file: {str(e)}')

def load_fake_real_news_data():
    """Load the fake or real news dataset and preprocess the data."""
    download_file(FAKE_REAL_NEWS_CSV_URL, 'fake_or_real_news.csv')
    data = pd.read_csv('fake_or_real_news.csv')
    data['fake'] = data['label'].apply(lambda x: 0 if x == "REAL" else 1)
    data = data.drop("label", axis=1)
    return data['text'], data['fake']

def train_fake_news_classifier(X, y):
    """Train a LinearSVC classifier for fake news detection."""
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_vectorized = vectorizer.fit_transform(X)
    classifier = LinearSVC()
    classifier.fit(X_vectorized, y)
    return classifier, vectorizer
def predict_fake_news(classifier, vectorizer, text):
    """Predict whether a given text is likely fake news."""
    vectorized_text = vectorizer.transform([text])  # Wrap the text in a list to create a sequence
    prediction = classifier.predict(vectorized_text)
    return prediction[0]

def calculate_fake_news_ratio(Xdata, classifier, vectorizer):
    """Calculate the ratio of fake news in a given dataset."""
    results = []
    for i in range(len(Xdata)):
        vectorized_text = vectorizer.transform([Xdata.values[i][0]])  # Access the text value from the DataFrame
        ratingarray = classifier.predict(vectorized_text)
        results.append(ratingarray[0])
    percentage = sum(results) / len(results)
    return percentage
def get_article_text(url):
    """Retrieve the article text from a given URL."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    article_body = soup.find('div', class_='article-body-commercial-selector')

    if article_body is not None:
        article_paragraphs = article_body.find_all('p')
        article_text = ' '.join([paragraph.text for paragraph in article_paragraphs])
        return article_text
    else:
        return None

def scrape_articles(url):
    """Scrape articles from a given URL and save them in a CSV file."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    article_links = soup.find_all('a', class_='u-faux-block-link__overlay')

    articles = []
    for link in article_links:
        article_url = link['href']
        article_text = get_article_text(article_url)
        if article_text is not None:
            articles.append([article_text])
            logger.info(f"Article saved: {article_url}")

    with open(ARTICLES_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(articles)
    logger.info(f"Articles saved in file: {ARTICLES_CSV_FILE}")

def main():
    # Download the necessary files
    download_file(UNIDENTIFIED_TEXT_URL, "unidentified.txt")
    download_file(GUARDIAN_ARTICLES_XLSX_URL, 'guardianarticles3.xlsx')
    download_file(SUN_ARTICLES_XLSX_URL, 'sunarticles2.xlsx')
    download_file(INFOWARS_ARTICLES_XLSX_URL, 'infowarsarticles2.xlsx')

    # Load the fake or real news data
    X, y = load_fake_real_news_data()

    # Train the fake news classifier
    classifier, vectorizer = train_fake_news_classifier(X, y)

    # Predict fake news for unidentified text
    with open("unidentified.txt", "r", encoding="utf-8") as f:
        text = f.read()
    rating1 = predict_fake_news(classifier, vectorizer, text)
    if rating1 == 1:
        print("Likely fake news.")
    else:
        print("Likely real news.")

    # Calculate fake news ratio for Guardian articles
    Xdata = pd.read_excel("guardianarticles3.xlsx")
    percentage = calculate_fake_news_ratio(Xdata, classifier, vectorizer)
    print(f"Ratio of fake news on this site: {percentage}")
    if percentage > 0.70:
        print("I would not recommend that you rely on this source. It seems to have a high proportion of fake news.")
    elif 0.3 <= percentage <= 0.7:
        print("This seems to have a mix of true and fake news.")
    else:
        print("This seems to be a mostly reliable site.")

    # Calculate fake news ratio for Sun articles
    Xdata = pd.read_excel("sunarticles2.xlsx")
    percentage = calculate_fake_news_ratio(Xdata, classifier, vectorizer)
    print(f"Ratio of fake news on this site: {percentage}")
    if percentage > 0.70:
        print("I would not recommend that you rely on this source. It seems to have a high proportion of fake news.")
    elif 0.3 <= percentage <= 0.7:
        print("This seems to have a mix of true and fake news.")
    else:
        print("This seems to be a mostly reliable site.")

    # Calculate fake news ratio for Infowars articles
    Xdata = pd.read_excel("infowarsarticles2.xlsx")
    percentage = calculate_fake_news_ratio(Xdata, classifier, vectorizer)
    print(f"Ratio of fake news on this site: {percentage}")
    if percentage > 0.70:
        print("I would not recommend that you rely on this source. It seems to have a high proportion offake news.")
    elif 0.3 <= percentage <= 0.7:
        print("This seems to have a mix of true and fake news.")
    else:
        print("This seems to be a mostly reliable site.")

    # Get text from a user-specified URL


    # Scrape articles from The Guardian and download the CSV file
    scrape_articles(GUARDIAN_URL)
    logger.info("Script execution completed successfully.")

if __name__ == '__main__':
    main()


