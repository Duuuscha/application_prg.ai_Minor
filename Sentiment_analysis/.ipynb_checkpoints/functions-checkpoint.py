import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=["positive", "negative"]):
    """
    function with all params needed to produce nice confusion matrix
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.grid(False)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

def cleaner(sentence):
    """
    gets rid of marks, punctuation and html tags
    """
    cleaned = re.sub(r'<.*?>', r' ', sentence) # gets rid of html
    cleaned = re.sub(r'[?|!|\'|\"|#|-]',r' ',cleaned) # types of marks
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned) # types of punctuation
    return cleaned


def collect_words(df,frame_column):
    """
    clear text in column of given data frame
    """
    stop_words = stopwords.words('english')
    porter = PorterStemmer()
    all_sent =[]

    for text in df[frame_column].values:
        sentences = nltk.sent_tokenize(text) # sentences for one review
        current_sentence=[] 
        for sentence in sentences: 
            cleaned = cleaner(sentence) 
            words = nltk.word_tokenize(cleaned)
            no_stops = [word for word in words if word.lower() not in stop_words] # remove stop words
            stems = [porter.stem(word.lower()) for word in no_stops] # create stem words
            current_sentence+=stems
        all_sent.append(" ".join(current_sentence))
    return all_sent

def helpful_review(x):
    """splits review/helpfulness to find how many of them were actually voted as not helpfull"""
    xsplit = x.split("/")
    if xsplit[1]=="":
        return True
    else:
        return (int(xsplit[0])/int(xsplit[1]))>=1   