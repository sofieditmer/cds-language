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
import argparse

# Data and nlp
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner"])
nlp.max_length = 67200352

# Vizualisation
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
    
    # Argument 1: number of topics 
    ap.add_argument("-t", "--num_topics",
                    type = int,
                    required = False,
                    help = "Define the number of topics you want",
                    default = 10)
    
    # Argument 2: the size of the chunks
    ap.add_argument("-c", "--chunk_size",
                    type = int,
                    required = False,
                    help = "Define the size of the chunks",
                    default = 10)
    
    # Argument 3: the number of passes
    ap.add_argument("-p", "--passes",
                    type = int,
                    required = False,
                    help = "Define the number of passes",
                    default = 10)
    
    # Argument 4: minimum count for bigrams
    ap.add_argument("-m", "--min_count",
                    type = int,
                    required = False,
                    help = "Define the minimum count for bigrams to occur to be included",
                    default = 3)
    
    # Argument 5: threshold
    ap.add_argument("-th", "--threshold",
                    type = int,
                    required = False,
                    help = "Define the threshold which determines which phrases to include and which to exlude",
                    default = 50)
    
    # Argument 6: iterations
    ap.add_argument("-i", "--iterations",
                    type = int,
                    required = False,
                    help = "Define the number of iterations through each document in the corpus",
                    default = 100)

    # Parse arguments
    args = vars(ap.parse_args())
    
    # Save input parameters
    num_topics = args["num_topics"]
    chunk_size = args["chunk_size"]
    passes = args["passes"]
    min_count = args["min_count"]
    threshold = args["threshold"]
    iterations = args["iterations"]
    
    # Create output directory
    if not os.path.exists(os.path.join("..", "out")):
        os.mkdir(os.path.join("..", "out"))
    
    # Start message to user
    print("\n[INFO] Initializing topic modeling on all Donald Trump tweets from May 2009 to June 2020...")
    
    # Load and prepare data
    print("\n[INFO] Loading and preprocessing data...")
    filename = os.path.join("..", "..", "data", "trumptweets.csv")
    data = load_data(filename)
    
    # Process data
    print("\n[INFO] Creating bigram and trigram models and performing lemmatization and part-of-speech-tagging...")
    processed_data = process_data(data, min_count, threshold)
    
    # Create bag of words
    print("\n[INFO] Creating dictionary and word corpus...")
    dictionary, corpus = create_dict_corpus(processed_data)
    
    # Create LDA model and compute perplexity and coherence scores
    print("\n[INFO] Creating LDA model...")
    lda_model, perplexity_score, coherence_score = create_lda(processed_data, dictionary, corpus, num_topics, chunk_size, passes, iterations)
    print("\n[INFO] Calculating perplexity and coherence scores...")
    print(f"Perplexity score: {perplexity_score}, Coherence score: {coherence_score}")
          
    # Create outputs
    outputs(lda_model, perplexity_score, coherence_score, corpus, processed_data)
    print("\n[INFO] A txt-file containing the topics has been saved to output directory...")
    print("\n[INFO] A dataframe showing the most dominant topic for each chunk has been saved to output directory...")
    print("\n[INFO] A dataframe showing the most contributing keywords for each topic has been saved to output directory...")

    # User message
    print("\n[INFO] Done! You have now performed topic modeling on all of Donald Trump tweets from May 2009 to June 2020. The results have been saved in the out directory.\n")
    
    
### FUNCTIONS USED WITHIN MAIN FUNCTION ###

def load_data(filename):
    """
    Load data with pandas
    """
    # Read data
    tweets_df = pd.read_csv(filename)
    
    # Select the relevant columns 
    tweets_df = tweets_df.loc[:, ("id", "content", "date")]
    
    return tweets_df

def process_data(data, min_count, threshold):
    """
    Create bigram and trigram models, and perform lemmatization and part-of-speech-tagging
    """
          
    # Create model of bigrams and trigrams 
    bigram = gensim.models.Phrases(data["content"], min_count = min_count, threshold = threshold)
    trigram = gensim.models.Phrases(bigram[data["content"]], threshold = threshold)
    
    # Fit the models to the data
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    # Lemmatize and part-of-speech tag
    processed_data = lda_utils.process_words(data["content"],
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
    dictionary = corpora.Dictionary(processed_data)
    
    # Create Corpus: Term Document Frequency
    corpus = [dictionary.doc2bow(text) for text in processed_data]
    
    return dictionary, corpus

def create_lda(processed_data, dictionary, corpus, num_topics, chunk_size, passes, iterations):
    """
    Create LDA model and compute perplexity and coherence scores
    """
    # Define and run LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=num_topics,
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
                                         dictionary=dictionary,
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
        f.writelines(f"Perplexity score: {perplexity_score}, Coherence score: {coherence_score}")
        f.writelines(f"\nTopics: {topics}")
    
    # Output 2: keywords for each topic #
          
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
    sent_topics_sorteddf.reset_index(drop=True, inplace=True)
    
    # Define columns in dataframe
    sent_topics_sorteddf.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]
    
    # Save dataframe to output-folder
    sent_topics_sorteddf.to_csv(os.path.join("..", "out", "topic_contributions.csv"), index = False)
          
# Define behaviour when called from command line
if __name__=="__main__":
    main()