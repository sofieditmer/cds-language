#!/usr/bin/env python

#This is a script that calculates collocates for a specific keyword by computing how often each word collocates with the keyword across a text corpus. A measure of the strength of association between the keyword and its collocates (MI) is calculated, and the results are saved as a single file containing the keyword, collocate, raw frequency, and the MI. 

"""
Calculate collocates for specific keyword in text corpus
Parameters:
    path: str <path-to-folder>
Usage:
    Assignment2_collocation.py -p <path-to-folder>
Example:
    $ python Assignment2_collocation.py -p data/100_english_novels/corpus
"""

# IMPORT LIBRARIES #
import os
import sys 
sys.path.append(os.path.join("..")) # enabling communication with home directory
import pandas as pd 
from pathlib import Path
import csv 
import re
import string
import numpy as np
import argparse

# DEFINE TOKENIZE() FUNTION # 

# First we need a tokenize() function that takes an input string and splits it into tokens (individual words) and then returns a list of the individual words

def tokenize(input_string):
    # Using the compile() function from the re module to search for all characters except for lowercase and uppercase letters and apostrophes.
    tokenizer = re.compile(r"[^a-zA-Z']+") 
    # Create a list of tokens (words) by splitting the input string at the compiling pattern defined above
    token_list = tokenizer.split(input_string)
    # Return the list of tokens (individual words)
    return token_list

# DEFINE MAIN FUNCTION # 
                                 
# Now we need a function that takes a path, a keyword, and a window size and calculates how often each word in the corpus collocates with the keyword across the corpus. It should then calculate the mutual information (MI) between the keyword and all collocates across the corpus. The MI is a measure of the strength of association between the keyword and the collocate in question. The function should return a dataframe with four columns: keyword, collocate, raw_frequency, and MI.

def main():
    
    # First I want to define the arguments that the function requires in order to be run from the command line 
    # I do this using the argparse module
 
    # Define function arguments 
    ap = argparse.ArgumentParser()
    # Argument 1: the first argument is the path to the corpus directory
    ap.add_argument("-p", "--path", required = True, help= "Path to directory of text corpus")
    # Argument 2: the second argument is the keyword
    ap.add_argument("-k", "--keyword", required = True, help= "Key/target word in lowercase letters")
    # Argument 3: the third argument is the window size
    ap.add_argument("-w", "--windowsize", required = True, help= "Window size in number of words")
    # Create a variable containing the argument parameters defined above
    args = vars(ap.parse_args())
    
    # Define path to text corpus
    input_path = args["path"]
    # Define keyword
    keyword = args["keyword"]
    # Define window size
    window_size = int(args["windowsize"])
    
    # Now we can move on to the actual function
    
    # Create empty list for all tokens across the corpus
    token_list_all = []
    #Create empty list of all collocates
    collocates_list = []
    # Create empty dataframe with the four columns
    data = pd.DataFrame(columns=["keyword", "collocate", "raw_frequency", "MI"])
    # Create empty u (keyword) variable 
    u = 0
    
# Create a loop that goes through each file in the corpus and does the following: reads the file, tokenizes it using the tokenize() function defined previously, appends the list of all tokens across the corpus with the tokens from each individual file, makes a list of indices in which the particular positions (index) of the keyword in the token list is found, and calculates u (the number of occurrences of the keyword in the token_list)
                                 
    for filename in Path(path).glob("*.txt"):
        with open (filename, "r", encoding = "utf-8") as file:
            text = file.read()
            # Create a list of tokens consisting of the tokens from each file
            token_list = tokenize(text.lower())
            # Create a list of all tokens from across the corpus by extending/appending the token list for each file
            token_list_all.extend(token_list)
            # Create a list of all occurrences of the keyword across the file
            indices = [index for index, x in enumerate(token_list) if x == keyword]
            # Calculate u which is the number of keyword occurrences, hence the length of the indices list
            u = u + len(indices)
            
# Now a loop within the loop above is created. This loop goes through the list of keyword occurrences (indices) and finds the word(s) before and after the keyword based on the window_size parameter. 
            for index in indices:
                # Define window_start
                window_start = max(0, index - window_size)
                # Define window_end
                window_end = index + window_size
                # Define the entire string: {word(s) before keyword} {keyword} {word(s) after keyword}
                keyword_string = token_list[window_start : window_end + 1] # + 1 is added to include the last index
                # Append the list of collocates with the keyword_string
                collocates_list.extend(keyword_string)
                # Remove the keyword so only the collocates appear in the list
                collocates_list.remove(keyword)
                
# A new loop is now created. This loop goes through a list of the unique collocates (only one occurrence per collocate) and calculates v (how often the collocate occurs), the raw frequency aka. O11 (how often the collocate occurs in the same context as the keyword), O12 (how often the keyword occurs without the collocate), O21 (how often the collocate occurs without the keyword), R1 (O11 + O12), C1 (O11 + O21), N (total number of words), E11 (R1*C1/N), and MI (mutual information) which is the strength of association between the keyword and the collocate
                                 
    # Create empty list of the unique collocates using the set() function
    unique_collocates = set(collocates_list)
    
    for collocate in unique_collocates:
        # Calculate v (number of occurrences of the collocate across the entire corpus)
        v = token_list_all.count(collocate)
        # Calculate O11 (number of occurrences of the collocate) 
        O11 = collocates_list.count(collocate)
        # Calculate O12 (how often the keyword occurs without the collocate)
        O12 = u - O11
        # Calculate O21 (how often the collocate occurs without the keyword)
        O21 = v - O11
        # Calculate R1 (the number of times the keyword occurs with any collocate within the window size)
        R1 = O11 + O12
        # Calculate C1 (the number of times the collocate appears across the whole corpus)
        C1 = O11 + O21
        # Calculate N (the total length of the corpus)
        N = len(token_list_all)
        # Calculate E11
        E11 = R1*C1/N
        # Calculate MI
        MI = np.log(O11/E11)
                                 
        # Append to the dataframe with the calculated parameters 
        data = data.append({"keyword": keyword, 
                     "collocate": collocate, 
                     "raw_frequency": O11,
                     "MI": MI}, ignore_index = True)
        
    # Sort by the MI values in descending order so the collocate with the highest MI appears at the top of the list
    data = data.sort_values("MI", ascending = False)  
    
    # Return the dataframe
    return data
                                 
# Now we have a dataframe consisting of four columns: keyword, collocate, raw_frequency, and MI

# APPLY THE FUNCTION TO THE 100 ENGLISH NOVELS CORPUS # 

# Now we can actually use the function on a text corpus consisting of 100 English novels

# Define path
path = os.path.join("..", "data", "100_english_novels", "corpus")

# Use the main() function with example keyword ("sunshine") and window size (2)
collocates_df = main(path, "sunshine", 2)

# Save the dataframe as a csv-file
collocates_df.to_csv("Collocates.csv", index = False)
                                 
# Define behaviour when called from command line
if __name__=="__main__":
    main()