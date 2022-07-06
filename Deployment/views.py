# importing the libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
import re
import pickle

# nltk imports
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet as swn



# bert imports
import transformers 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.model_selection import train_test_split

# instantiating spacy's pretrained model
nlp = spacy.load("en_core_web_sm")


# main function
def main():
    df = load_data()
    # login page
    st.title("The Bold and the Beautiful")

    # setting header container
    header = st.container()

    # setting columns within the header container
    with header:
        col1, col2 = st.columns([1,6])

        with col1:
            st.image("amazon_icon.png", width=100,)

        with col2:
            st.markdown("<h1 style='text-align: left; color: darkorange;'>Amazon Product Reviews</h1>", unsafe_allow_html=True)


    # navigation bar
    menu = ["Home", "Explore", "Review Analysis"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.markdown("<h1 style='text-align: left; color: darkorange;'>Aspect Based Sentiment Analysis</h1>", unsafe_allow_html=True)
        # Adding spaces between the header and image
        st.write()
        st.write()
        c1,c2,c3 = st.columns([1,5,1])

        with c1:
            st.write()
        with c2:

            imagemain = Image.open("absa.jpeg")
            st.image(imagemain, caption="ABSA in action", width =450)
            st.markdown("##")
        with c3:
            st.write('')

        c1,c2,c3 = st.columns([1,5,1])
        with c1:
            st.write('')
        with c2:
            st.markdown('''*Aspect-Based Sentiment Analysis (ABSA) is a type of text analysis that categorizes 
            opinions by aspect and identifies the sentiment related to each aspect. It can be used to analyze customer 
            feedback by associating specific sentiments with different aspects of a product or service.*''')
        with c3:
            st.write('')
        

        c1,c2,c3 = st.columns([3,5,1])
        with c1:
            st.write('')
        with c2:
            st.markdown('''*Sources*''')
        with c3:
            st.write('')

    elif choice == "Explore":
        st.markdown("<h2 style='text-align: left; color: darkorange;'>Data Exploration</h1>", unsafe_allow_html=True)
        st.markdown("Overall reviews for a brand")
        user_input = st.text_input("Enter the file", "absa.csv")
        df = pd.read_csv(user_input)
        x_axis = st.selectbox("Choose a brand", df["brand"].unique(), index=3)
        visualize_data(df, x_axis)

    elif choice == "Review Analysis":
        user_input = st.text_area("Enter a review", "I love this ring! This is a ring I have not taken off since I got it. It fits perfectly and because of its sturdy material and great design, I don't have to worry about it getting damaged or dirty even when I'm out in the gardens. It's very comfortable to wear and easy to clean. I definitely recommend it for anyone looking for a ring that can handle a lot of wear and tear.")
        analyzer(user_input)
        ABSA(user_input)

@st.cache
def load_data():
    df = pd.read_csv("absa.csv")
    return df

def visualize_data(df, x_axis):
    sns.barplot(x = 'brand', y = 'overall', data=df[df['brand']==x_axis])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    

def analyzer(data):
  columns = ['Review', 'Sentiment']
  reviews = list(filter(None, data.strip().split('.')))
  reviews = [remove_special(review) for review in reviews]
  reviews = [POSTag(review) for review in reviews]
  reviews = [lemmatize(review) for review in reviews]
  sent_model = pickle.load(open('bert.sav', 'rb'))
  sent_pred_num = [SentimentScore(text) for text in reviews]
  sent_pred = [getSentiment(s) for s in sent_pred_num]
  d = {'Review':reviews, 'Sentiment':sent_pred}
  final_df = pd.DataFrame(d, columns=columns)
  st.write(final_df)


############# ABSA #################
def ABSA(data):
  columns = ['Reviews', 'Aspect']
  reviews = list(filter(None, data.strip().split('.')))
  reviews = [remove_special(review) for review in reviews]
  reviews = [POSTag(review) for review in reviews]
  reviews = [lemmatize(review) for review in reviews]
  aspects = []
  i = 0
  for i, sentence in enumerate(data):
    doc = nlp(sentence)
    descriptive_term = ''
    target = ''
    for token in doc:
      if (token.pos_ == 'NOUN'):
        target = token.text
      if token.pos_ == 'ADJ':
        prepend = ''
        for child in token.children:
          if child.pos_ != 'ADV':
            continue
          prepend += child.text + ' '
        descriptive_term = prepend + token.text
    aspects.append({'aspect': target})
  aspects = {a:[t] for a,t in aspects[0].items()}  
  d = {'Review':reviews, 'Aspect':aspects}
  final_df = pd.DataFrame(d, columns=columns)
  st.write(final_df)


#################### start of preprocessing ########################

def remove_special(text):
    # Removing special characters
    text = re.sub('[^a-zA-Z]+', ' ', text)
    return text

# Tokenizing and POS tagging for lemmatization
def POSTag(text):
  tagged = pos_tag(word_tokenize(text))
  # List that will contain final results
  tagged_new = []
  # Dictionary to covert POS tags to WordNet tags for lemmatization
  pos_dict =  {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}
  # Removing stop words
  for word, tag in tagged:
    if word.lower() not in set(stopwords.words('english')):
      tagged_new.append(tuple([word, pos_dict.get(tag[0])]))
  return tagged_new

# Lemmatization function
def lemmatize(tagged_data):
  lemmatizer = WordNetLemmatizer()
  lemmatized = ' '
  for word, pos in tagged_data:
    if not pos:
      lemma = word
      lemmatized = lemmatized + ' ' + lemma
    else:
      lemma = lemmatizer.lemmatize(word, pos=pos)
      lemmatized = lemmatized + ' ' + lemma
  return lemmatized
    
################ end of preprocessing ##################



################ BERT ##################
# Tokenizer

# tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
# model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


# Sentiment scores
def SentimentScore(text):
  score = TextBlob(text).sentiment.polarity
  if score >= 0.0:
    return 1;
  else:
    return 0;

def getSentiment(number):
  if number == 0:
    return "Negative"
  elif number == 1:
    return "Neutral"
  else:
    return "Positive"

        


if __name__ == '__main__':
    main()








