#!/usr/bin/env python

"""
This script performs topic modeling on all tweets by Donald Trump from May 2009 to June 2020.

Usage:
    $ python lda.py
"""

### DEPENDENCIES ###

# Standard library
import sys,os
sys.path.append(os.path.join(".."))
from pprint import pprint

# Argparse
import argparse

# Numpy
import numpy as np

# Data and nlp
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner"])
nlp.max_length = 67200352 # increasing the maximum length

# Vizualisation
import pyLDAvis.gensim
import seaborn as sns
from matplotlib import rcParams
# Figure size in inches
rcParams['figure.figsize'] = 20,10

# LDA tools
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from utils import lda_utils

# Warnings
import logging, warnings
warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

### MAIN FUNCTION ###

def main():
    
    ### ARGPARSE ###
    
    # Initialize ArgumentParser class
    ap = argparse.ArgumentParser()
    
    # Argument 1: the size of the chunks
    ap.add_argument("-c", "--chunk_size",
                    type = int,
                    required = False,
                    help = "Define the size of the chunks",
                    default = 10)
    
    # Argument 2: the number of passes
    ap.add_argument("-p", "--passes",
                    type = int,
                    required = False,
                    help = "Define the number of passes",
                    default = 10)
    
    # Argument 3: minimum count for bigrams
    ap.add_argument("-m", "--min_count",
                    type = int,
                    required = False,
                    help = "Define the minimum count for bigrams to occur to be included",
                    default = 3)
    
    # Argument 4: threshold
    ap.add_argument("-th", "--threshold",
                    type = int,
                    required = False,
                    help = "Define the threshold which determines which phrases to include and which to exlude",
                    default = 50)
    
    # Argument 5: iterations
    ap.add_argument("-i", "--iterations",
                    type = int,
                    required = False,
                    help = "Define the number of iterations through each document in the corpus",
                    default = 100)
    
    # Argument 6: rolling mean size
    ap.add_argument("-r", "--rolling_mean",
                    type = int,
                    required = False,
                    help = "Define the rolling mean",
                    default = 50)

    # Parse arguments
    args = vars(ap.parse_args())
    
    # Save input parameters
    chunk_size = args["chunk_size"]
    passes = args["passes"]
    min_count = args["min_count"]
    threshold = args["threshold"]
    iterations = args["iterations"]
    rolling_mean = args["rolling_mean"]
    
    # Create output directory
    if not os.path.exists(os.path.join("..", "out")):
        os.mkdir(os.path.join("..", "out"))
        
    # Create directory for vizualization
    if not os.path.exists(os.path.join("..", "viz")):
        os.mkdir(os.path.join("..", "viz"))
    
    # Start message to user
    print("\n[INFO] Initializing topic modeling on all Donald Trump tweets from May 2009 to June 2020...")
    
    # Load and prepare data
    print("\n[INFO] Loading and preprocessing data...")
    filename = os.path.join("..", "..", "data", "trumptweets.csv")
    data = load_data(filename, chunk_size)
    
    # Process data
    print("\n[INFO] Creating bigram and trigram models and performing lemmatization and part-of-speech-tagging...")
    processed_data = process_data(data, min_count, threshold)
    
    # Create bag of words
    print("\n[INFO] Creating dictionary and word corpus...")
    id2word, corpus = create_dict_corpus(processed_data)
    
    # Estimate the optimal number of topics
    print("\n[INFO] Finding the optimal number of topics...")
    optimal_n_topics = find_optimal_n_topics(processed_data, corpus, id2word)
    print(f"\nThe optimal number of topics is {optimal_n_topics}")  
    
    # Create LDA model and compute perplexity and coherence scores
    print("\n[INFO] Creating LDA model...")
    lda_model, perplexity_score, coherence_score = create_lda(processed_data, id2word, corpus, optimal_n_topics, chunk_size, passes, iterations)
    print("\n[INFO] Calculating perplexity and coherence scores...")
    print(f"\nPerplexity score: {perplexity_score}, Coherence score: {coherence_score}")
          
    # Create outputs
    outputs(lda_model, perplexity_score, coherence_score, corpus, processed_data)
    print("\n[INFO] A txt-file containing the topics has been saved to output directory...")
    print("\n[INFO] A dataframe showing the most dominant topic for each chunk has been saved to output directory...")
    print("\n[INFO] A dataframe showing the most contributing keywords for each topic has been saved to output directory...")
    
    # Create visualization: topics over time with rolling mean
    print("\n[INFO] Creating visualization of topics over time...")
    visualize(processed_data, rolling_mean, lda_model, corpus)

    # User message
    print("\n[INFO] Done! You have now performed topic modeling on all of Donald Trump tweets from May 2009 to June 2020. The results have been saved in the out and viz directories.\n")
    
    
### FUNCTIONS USED WITHIN MAIN FUNCTION ###

def load_data(filename, chunk_size):
    """
    Load data and make chunks of 10 tweets.
    """
    # Read data
    tweets_df = pd.read_csv(filename,
                            lineterminator = "\n")
    
    # Select the relevant columns 
    tweets_df = tweets_df.loc[:, ("id", "content", "date")]
    
    # Create empty list for chunks of tweets
    chunks = []
    
    # Loop through the tweets and create chunks of 10 tweets
    for i in range(0, len(tweets_df["content"]), chunk_size):
        chunks.append(' '.join(tweets_df["content"][i:i+chunk_size]))
    
    return chunks

def process_data(data, min_count, threshold):
    """
    Create bigram and trigram models, and perform lemmatization and part-of-speech-tagging
    """
          
    # Create model of bigrams and trigrams 
    bigram = gensim.models.Phrases(data, min_count = min_count, threshold = threshold)
    trigram = gensim.models.Phrases(bigram[data], threshold = threshold)
    
    # Fit the models to the data
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    # Lemmatize and part-of-speech tag
    processed_data = lda_utils.process_words(data,
                                         nlp, 
                                         bigram_mod, 
                                         trigram_mod, 
                                         allowed_postags=["NOUN"])
    return processed_data

def create_dict_corpus(processed_data):
    """
    Create dictionary and corpus
    """
    # Create Dictionary
    id2word = corpora.Dictionary(processed_data)
    
    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in processed_data]
    
    return id2word, corpus

def find_optimal_n_topics(processed_data, corpus, id2word):
    """
    Run model multiple times with different numbers of topics and find the optimal number based on the maximum coherence value.
    """
    
    # Run model multiple times
    model_list, coherence_values = lda_utils.compute_coherence_values(texts = processed_data,
                                                                      corpus = corpus,
                                                                      dictionary = id2word,
                                                                      start = 5,
                                                                      limit = 40,
                                                                      step = 5)
    # Find the maximum coherence value
    max_coherence = np.argmax(coherence_values)
    
    # Find the number of topics corresponding to the maximum coherence value
    optimal_n_topics = model_list[max_coherence].num_topics
    
    return optimal_n_topics

def create_lda(processed_data, id2word, corpus, optimal_n_topics, chunk_size, passes, iterations):
    """
    Create LDA model and compute perplexity and coherence scores
    """
    # Define and run LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=optimal_n_topics,
                                           random_state=100,
                                           chunksize=chunk_size,
                                           passes=passes,
                                           iterations=iterations,
                                           per_word_topics=True,
                                           minimum_probability=0.0)
    
    # Calculate perplexity score
    perplexity_score = lda_model.log_perplexity(corpus)
    
    # Calculate coherence score
    coherence_model_lda = CoherenceModel(model=lda_model,
                                         texts=processed_data,
                                         dictionary=id2word,
                                         coherence='c_v')
    
    coherence_score = coherence_model_lda.get_coherence()
          
    return lda_model, perplexity_score, coherence_score
          
def outputs(lda_model, perplexity_score, coherence_score, corpus, processed_data):
    """
    Create outputs that are saved in the output folder. 
    """
    
  # Output 1: topics + perplexity and coherence scores #
    
    # Get the topics
    topics = lda_model.print_topics()
    
    # Write txt-file containing the topics + perplexity and coherence scores
    with open("../out/topics.txt", "w+") as f:
        # Print perplexity and coherence scores
        f.writelines(f"Perplexity score: {perplexity_score}, Coherence score: {coherence_score} \n")
        # Print topics
        f.writelines(f"\nOverview of topics: \n {topics}")
    
  # Output 2: most dominant topic per chunk #
          
    # Find keywords for each topic
    df_topic_keywords = lda_utils.format_topics_sentences(ldamodel=lda_model,
                                                          corpus=corpus,
                                                          texts=processed_data)
    # Find the most dominant topic per chunk
    df_dominant_topic = df_topic_keywords.reset_index()
    df_dominant_topic.columns = ['Chunk_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    
    # Save dataframe to output folder
    df_dominant_topic.to_csv(os.path.join("..", "out", "dominant_topic.csv"), index = False)

  # Output 3: most contributing topic # 
          
    # Display setting to show more characters in column
    pd.options.display.max_colwidth = 100
    
    # Create dataframe
    sent_topics_sorteddf = pd.DataFrame()

    # Group keywords by the most dominant topic
    sent_topics_outdf_grpd = df_topic_keywords.groupby('Dominant_Topic')
    
    # Compute how much each topic contribtues in percentage
    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf = pd.concat([sent_topics_sorteddf, grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], axis=0)
    
    # Reset index
    sent_topics_sorteddf.reset_index(drop=True, inplace=True)
    
    # Define columns in dataframe
    sent_topics_sorteddf.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]
    
    # Save dataframe to output-folder
    sent_topics_sorteddf.to_csv(os.path.join("..", "out", "topic_contributions.csv"), index = False)

def visualize(processed_data, rolling_mean, lda_model, corpus):
    """
    Visualize topics over time with seaborn
    """
    # Create viz object 
    viz = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary = lda_model.id2word)
    
    # Create list of values. The first entry is the topic, and the second entry is how much it contributes (percentage)
    values = list(lda_model.get_document_topics(corpus))
          
    # Split the values and keep only the values per topic
    split = []
    for entry in values:
        topic_prevelance = []
        for topic in entry:
            topic_prevelance.append(topic[1])
        split.append(topic_prevelance)
    
    # Create document-topic matrix
    matrix = pd.DataFrame(map(list,zip(*split)))
    
    # Create plot with rolling mean
    lineplot = sns.lineplot(data=matrix.T.rolling(rolling_mean).mean())
    fig = lineplot.get_figure()
    fig.savefig("../viz/topics_over_time.jpg")

# Define behaviour when called from command line
if __name__=="__main__":
    main()