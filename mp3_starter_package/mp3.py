# Starter code for CS 165B HW3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn import svm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import numpy as np
import sklearn
import pandas as pd
#import torch

def run_train_test(training_data, training_labels, testing_data):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.

    Inputs:
        training_data: List[string]
        training_label: List[string]
        testing_data: List[string]

    Output:
        testing_prediction: List[string]
    Example output:
    return ['NickLouth']*len(testing_data)
    """

    #TODO implement your model and return the prediction

    vectorizer = TfidfVectorizer()
    vectorizer.fit(training_data)
    Encoder = LabelEncoder()
    Train_y = Encoder.fit_transform(training_labels)
    try1 = training_data[0]
    filtered_doc =[]
    stop_words = set(stopwords.words('english'))
    porter = PorterStemmer()
    for i in training_data:
        tokens = word_tokenize(i)
        words = [word for word in tokens if word.isalpha()]
        stemmed_words = [porter.stem(word) for word in tokens]
        filtered_tokens = [w for w in stemmed_words if not w in stop_words]
        filtered = (" ").join(filtered_tokens)
        print(filtered)
        filtered_doc.append(filtered)
    

    tfidftrainx = vectorizer.transform(filtered_doc)
    #print(tfidftrainx[0])
    tfidftestx = vectorizer.transform(testing_data)
    SVM = svm.SVC(C=1.0,kernel='linear',degree=3, gamma='auto')
    SVM.fit(tfidftrainx,Train_y)
    predictions_Svm = SVM.predict(tfidftestx)
    return Encoder.inverse_transform(predictions_Svm)
    