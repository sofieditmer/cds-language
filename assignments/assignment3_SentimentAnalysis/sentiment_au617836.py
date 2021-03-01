#!/usr/bin/env python

# This script calculates the sentiment score for every headline in the data. It creates and saves a plot of sentiment over time with a 1-week rolling average and a 1-month rolling average. Both plots have clear balues on the x-axis as well as title and x- and y-labels.

"""
Calculate sentiment score for headlines and create plots of rolling averages
Parameters:
    path: str <path-to-folder>
    output_path: str <path-for-output-file>
Usage:
    Assignment3_sentiment.py -p <path-to-folder>
Example:
    $ python Assignment3_sentiment.py -p data/abcnews-date-text.csv -o --output_path outputs/
"""

# Import dependencies
import os
import spacy
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Initialize spaCy by creating a spaCy object
nlp = spacy.load("en_core_web_sm")

# Initialize spacytextblob and add it to the pipeline
from spacytextblob.spacytextblob import SpacyTextBlob
spacy_text_blob = SpacyTextBlob()
nlp.add_pipe(spacy_text_blob)


# Define main function
def main():
    
    # First I want to define the arguments that the function requires in order to be run from the command line 
    # I do this using the argparse module
 
    # Define function arguments 
    ap = argparse.ArgumentParser()
    # Argument 1: the path to the corpus directory
    ap.add_argument("-p", "--path", required = True, help= "Path to directory of news headlines")
    # Argument 2: the output path (where you want the saved files to be located)
    ap.add_argument("-o", "--output_path", required = True, help = "Path to directory for output file")
    # Create a variable containing the argument parameters defined above
    args = vars(ap.parse_args())
    
    # Define path to input directory
    input_path = args["path"]
    
    # Define output path
    output_directory = args["output_path"]
    # Create output directory if it does not exist
    if not os.path.exists("outputs"):
        os.mkdir("outputs")

    # Read file
    data = os.path.join(input_path)
    data_df = pd.read_csv(data)
    
    
    ## CALCULATE SENTIMENT SCORE ##
    print("Calculating sentiment scores...")
    
    # Create empty list that we can append to in the loop
    sentiment_scores = []
    
    # Loop through each headline and calculate the sentiment score
    for headline in nlp.pipe(data_df["headline_text"], batch_size = 500):
        # Caclulate sentiment score
        sentiment = headline._.sentiment.polarity
        # Append to list
        sentiment_scores.append(sentiment)
        
    # Create a new column in the dataframe containing the sentiment scores using the insert() function
    data_df.insert(len(data_df.columns), "sentiment", sentiment_scores)
    # Specify output path
    output_path = os.path.join("outputs", "Headlines_Sentiment_Scores.csv")
    # Save dataframe
    data_df.to_csv(output_path, index = False)
    
    ## CALCULATE ROLLING AVERAGES ## 
    print("Calculating rolling mean averages and generating plots...")
    
    # In order to calcualte the rolling averages, we need the dates as the index of the dataframe
    # Create a new dataframe, convert the dates to datetime format and make them the index
    rolling_data = pd.DataFrame({"sentiment": sentiment_scores}, 
                            index = pd.to_datetime(data_df["publish_date"], format='%Y%m%d', errors='ignore'))
    
    # Calcuate sentiment with a 1-week rolling average
    smoothed_sentiment_week = rolling_data.sort_index().rolling("7d").mean()
    
    # Plot 1-week rolling average
    plt.figure() # create plot figure
    plt.title("Sentiment over time with a 1-week rolling average") # plot title
    plt.xlabel("Date") # x-label
    plt.xticks(rotation=45) # rotate the x-labels so they do not overlap
    plt.ylabel("Sentiment score") # y-label
    plt.plot(smoothed_sentiment_week, label = "One week rolling average") # plot
    plt.legend(loc = "upper right") # legend
    plt.savefig(os.path.join(output_directory, "smoothed_sentiment_week.png"), bbox_inches='tight') # save plot
    
    # Calcuate sentiment with a 1-month rolling average
    smoothed_sentiment_month = rolling_data.rolling("30d").mean()
    
    # Plot 1-month rolling average
    plt.figure() # create plot figure
    plt.title("Sentiment over time with a 1-month rolling average") # plot title
    plt.xlabel("Date") # x-label
    plt.xticks(rotation=45) # rotate the x-labels so they do not overlap
    plt.ylabel("Sentiment score") # y-label
    plt.plot(smoothed_sentiment_month, label = "One month rolling average") # plot
    plt.legend(loc = "upper right") # legend
    plt.savefig(os.path.join(output_directory,"smoothed_sentiment_month.png"), bbox_inches='tight') # save plot
    
## DONE ##
print("A CSV-file containing sentiment scores for each headline as well as plots containing 1-week and 1-month rolling averages are now saved in the output directory")
    
# Define behaviour when called from command line
if __name__=="__main__":
    main()
  