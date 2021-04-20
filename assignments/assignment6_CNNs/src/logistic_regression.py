#!/usr/bin/env python

"""
This script creates a baseline logistic regression model trained on the dialogue of all Game of Thrones 8 sesasons to predict which season a given line is from. This model can be used as a means of evaluating how a deep learning model performs. 
"""

### DEPENDENCIES ###

# system tools
import os
import sys
sys.path.append(os.path.join(".."))

# pandas, numpy, gensim, argparse
import pandas as pd
import numpy as np
import gensim.downloader
import argparse

# import my classifier utility functions
import utils.classifier_utils as clf

# Machine learning stuff
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit # cross-validation
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Embedding, 
                                     Flatten, GlobalMaxPool1D, Conv1D)
from tensorflow.keras.optimizers import SGD, Adam # optimization algorithms
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import L2 # regularization

# matplotlib
import matplotlib.pyplot as plt

### MAIN FUNCTION ###

def main():
    
    ### ARGPARSE ###
    
    # Initialize ArgumentParser class
    ap = argparse.ArgumentParser()
    
    # Argument 1: Path to training data
    ap.add_argument("-t", "--train_data",
                    type = str,
                    required = False,
                    help = "Path to the training data",
                    default = "../../data/Game_of_Thrones_Script.csv")

  
    # Parse arguments
    args = vars(ap.parse_args())
    
    # Save input parameters
    train_data = args["train_data"]

    # Create output directory
    if not os.path.exists(os.path.join("..", "out")):
        os.mkdir(os.path.join("..", "out"))

    # User message
    print("\n[INFO] Initializing the construction of a logistic regression classifier...")
    
    # Load data
    print("\n[INFO] Loading and preprocessing data...")
    vectorizer, sentences, labels, X_train_feats, X_test_feats, y_train, y_test = load_preprocess_data(train_data)
    
    # Logistic regression classifier
    print("\n[INFO] Building logistic regression classifier, training it on the data, and evaluating it...")
    lr_classifier(X_train_feats, X_test_feats, y_train, y_test)
    
    # Cross-validation
    print("\n[INFO] Performing cross-validation...")
    cross_validation(sentences, vectorizer, labels)
    
    # User message
    print("\n[INFO] Done! You have now defined and trained a logistic regression classifier. The results can be found in the output directory.\n")
    
### FUNCTIONS USED WITHIN MAIN FUNCTION ###
        
def load_preprocess_data(train_data):
    """
    Loads the data, creates X_train, X_test, y_train, and y_test, and uses CountVectorizer to vectorize the training data. 
    """
    # Load data
    data = pd.read_csv(train_data, lineterminator = "\n")
        
    # Take the relevant columns
    data = data.loc[:, ("Season", "Sentence")]
        
    # Extract dialogue as training data
    sentences = data['Sentence'].values
        
    # Extract season as labels
    labels = data['Season'].values
        
    # Create training and test split using sklearn
    X_train, X_test, y_train, y_test = train_test_split(sentences,
                                                        labels,
                                                        test_size=0.25, # 25% as test data
                                                        random_state=42)
    
    # Intialize count vectorizer with default parameters
    vectorizer = CountVectorizer()
        
    # Fit vectorizer to training and test data
    X_train_feats = vectorizer.fit_transform(X_train)
    X_test_feats = vectorizer.transform(X_test)
        
    return vectorizer, sentences, labels, X_train_feats, X_test_feats, y_train, y_test
        
def lr_classifier(X_train_feats, X_test_feats, y_train, y_test):
    """
    Creates logistic regression classifier as a simple benchmark.
    """
        
    # Fitting the classifier
    classifier = LogisticRegression(random_state=42, max_iter = 1000).fit(X_train_feats, y_train)
       
    # Extract predictions
    y_pred = classifier.predict(X_test_feats)
        
    # Evaluate model
    classifier_metrics = metrics.classification_report(y_test, y_pred)
        
    # Print to console
    print(classifier_metrics)
        
    # Save in output directory
    with open("../out/lr_classification_report.txt", 'w', encoding='utf-8') as f:
        f.writelines(metrics.classification_report(y_test, y_pred))
        
    # Plot results as a heat map and save to output directory
    clf.plot_cm(y_test, y_pred, normalized=True)
    plt.savefig("../out/lr_heatmap.png")
        
    return None
        
def cross_validation(sentences, vectorizer, labels):
    """
    Performs cross-validation and saves results in output directory.
    """
      
    # Vectorize the sentences
    X_vect = vectorizer.fit_transform(sentences)
       
    # Intialize cross-validation
    title = "Learning Curves (Logistic Regression)"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
        
    # Run cross-validation
    model = LogisticRegression(random_state=42, max_iter = 1000)
        
    # Plot learning curves and save in output directory
    clf.plot_learning_curve(model, title, X_vect, labels, cv=cv, n_jobs=4)
    plt.savefig("../out/lr_cross_validation_results.png")
        
    return None
        
# Define behaviour when called from command line
if __name__=="__main__":
    main()