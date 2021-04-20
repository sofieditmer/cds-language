#!/usr/bin/env python

"""
This script trains a deep learning CNN model on the dialogue from all 8 Game of Thrones seasons.
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
from contextlib import redirect_stdout

# import my classifier utility functions
import utils.classifier_utils as clf

# Import utility functions for this assignment
import utils.ass6_utils as utils

# Machine learning stuff
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit # cross-validation
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer

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
    
    # Argument 2: Number of epochs
    ap.add_argument("-e", "--n_epochs",
                    type = int,
                    required = False, 
                    help = "The number of epochs",
                    default = 20)
    
    # Argument 3: Batch size 
    ap.add_argument("-b", "--batch_size",
                    type = int,
                    required = False, 
                    help = "The batch size",
                    default = 10)
    
    # Argument 4: Size of test split
    ap.add_argument("-ts", "--test_split",
                    type = int,
                    required = False,
                    help = "The size of the test split, e.g. 0.25",
                    default = 0.25)

    # Parse arguments
    args = vars(ap.parse_args())
    
    # Save input parameters
    train_data = args["train_data"]
    n_epochs = args["n_epochs"]
    batch_size = args["batch_size"]
    test_size = args["test_split"]
                    
    # Create output directory
    if not os.path.exists(os.path.join("..", "out")):
        os.mkdir(os.path.join("..", "out"))

    # User message
    print("\n[INFO] Initializing the construction of a CNN model...")
    
    # Load and preprocess data
    print("\n[INFO] Loading and preprocessing the data...")
    X_train, X_test, vectorizer, y_train, y_test, sentences, labels, X_train_feats, X_test_feats = load_preprocess_data(train_data, test_size)
    tokenizer, vocab_size, X_train_toks, X_test_toks = word_embeddings(X_train, X_test)
    maxlen, X_train_pad, X_test_pad = padding(X_train_toks, X_test_toks)
    
    # Create, train, and evaluate CNN model 1
    print("\n[INFO] Defining the first CNN model, training it, and evaluating its performance...")
    cnn_model_1(vocab_size, maxlen, X_train_pad, y_train, X_test_pad, y_test, n_epochs, batch_size)
    
    # Create, train, and evaluate CNN model 2
    print("\n[INFO] Defining the second CNN model, training it, and evaluating its performance...")
    cnn_model_2(vocab_size, maxlen, X_train_pad, y_train, X_test_pad, y_test, n_epochs, batch_size)
    
    # User message
    print("\n[INFO] Done! You have now defined and trained two deep learning CNN models. The results can be found in the output directory.\n")
    
### FUNCTIONS USED WITHIN MAIN FUNCTION ###
        
def load_preprocess_data(train_data, test_size):
    """
    Loads the data, creates X_train, X_test, y_train, and y_test, uses CountVectorizer to vectorize the training data, and uses LabelBinarizer to binarize the data labels.  
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
                                                        test_size=test_size,
                                                        random_state=42)
    
    # Intialize count vectorizer with default parameters
    vectorizer = CountVectorizer()
        
    # Fit vectorizer to training and test data
    X_train_feats = vectorizer.fit_transform(X_train)
    X_test_feats = vectorizer.transform(X_test)
    
    # Binarize labels
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
        
    return X_train, X_test, vectorizer, y_train, y_test, sentences, labels, X_train_feats, X_test_feats

def word_embeddings(X_train, X_test):
    """
    Creates word embeddings with tf.keras.Tokenizer()
    """
    # Initialize tokenizer
    tokenizer = Tokenizer(num_words=5000) # vocabulary = 5000 words
    
    # Fit to raw training data
    tokenizer.fit_on_texts(X_train)
    
    # Use the tokenizer to create sequences of tokens
    X_train_toks = tokenizer.texts_to_sequences(X_train)
    X_test_toks = tokenizer.texts_to_sequences(X_test)
    
    # Overall vocabulary size
    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
    
    return tokenizer, vocab_size, X_train_toks, X_test_toks

def padding(X_train_toks, X_test_toks):
    """
    Pads the documents to be of equal length. 
    """
    # Define max length for a doc
    maxlen = 100
    
    # Pad training data to max length
    X_train_pad = pad_sequences(X_train_toks,
                                padding='post', # post = puts the pads at end of the sequence. Sequences can be padded "pre" or "post"
                                maxlen=maxlen)
    
    # Pad testing data to max length
    X_test_pad = pad_sequences(X_test_toks,
                               padding='post',
                               maxlen=maxlen)
    
    return maxlen, X_train_pad, X_test_pad

def cnn_model_1(vocab_size, maxlen, X_train_pad, y_train, X_test_pad, y_test, n_epochs, batch_size):
    """
    Creates CNN model architecture.
    """
    # Define embedding dimension
    embedding_dim = 100
    
    # Define L2 regularizer. The smaller the value, the more regularization
    l2 = L2(0.0001)
    
    # Intialize sequential model
    model = Sequential()
    
    # Add embedding layer that converts the numerical representations of the sentences into a dense, embedded representation
    model.add(Embedding(input_dim = vocab_size,
                        output_dim = embedding_dim,
                        input_length = maxlen))    
    
    # Add convolutional layer
    model.add(Conv1D(256, 5,
                     activation = 'relu',
                     kernel_regularizer = l2)) # L2 regularization 
    
    # Global max pooling
    model.add(GlobalMaxPool1D())
    
    # Add dense layer
    model.add(Dense(128, activation = 'relu',
                    kernel_regularizer = l2))
    
    # Add dense layer with 8 nodes; one for each season 
    model.add(Dense(8,
                    activation = 'softmax')) # we use softmax because it is a categorical classification problem
    
    # Compile model
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    
    # Print model summary
    model.summary()
    
    # Save model summary
    with open("../out/model_1_summary.txt", "w") as f:
        with redirect_stdout(f):
            model.summary()
            
    # Train model
    history = model.fit(X_train_pad, y_train,
                        epochs = n_epochs,
                        verbose = False,
                        validation_data = (X_test_pad, y_test),
                        batch_size = batch_size)
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_train_pad, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    
    # plot
    utils.plot_history(history, epochs = n_epochs)
    plt.savefig("../out/model_1_history.png")
    
def cnn_model_2(vocab_size, maxlen, X_train_pad, y_train, X_test_pad, y_test, n_epochs, batch_size):
    """
    Creates CNN model architecture.
    """
    
    # Define embedding dimension
    embedding_dim = 100
    
    # Define L2 regularizer. The smaller the value, the more regularization
    l2 = L2(0.0001)
    
    # Intialize sequential model
    model = Sequential()
    
    # Add embedding layer that converts the numerical representations of the sentences into a dense, embedded representation
    model.add(Embedding(input_dim = vocab_size,
                        output_dim = embedding_dim,
                        input_length = maxlen))    
    
    # Add convolutional layer
    model.add(Conv1D(256, 5,
                     activation = 'relu',
                     kernel_regularizer = l2)) # L2 regularization 
    
    # Global max pooling
    model.add(GlobalMaxPool1D())
    
    # Add dense layer
    model.add(Dense(128, activation = 'relu',
                    kernel_regularizer = l2))
    
    # Add dense layer with 8 nodes; one for each season 
    model.add(Dense(8,
                    activation = 'softmax')) # we use softmax because it is a categorical classification problem
    
    # Compile model
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    
    # Print model summary
    model.summary()
    
    # Save model summary
    with open("../out/model_2_summary.txt", "w") as f:
        with redirect_stdout(f):
            model.summary()
            
    # Train model
    history = model.fit(X_train_pad, y_train,
                        epochs = n_epochs,
                        verbose = False,
                        validation_data = (X_test_pad, y_test),
                        batch_size = batch_size)
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_train_pad, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    
    # plot
    utils.plot_history(history, epochs = n_epochs)
    plt.savefig("../out/model_2_history.png")
    
# Define behaviour when called from command line
if __name__=="__main__":
    main()