#!/usr/bin/env python
"""
For any given weighted edgelist of named entities based on document co-occurence, create a network visualization and save that in viz folder, create dataframe showing degree, betweenness, and eigenvector centrality for each node, and save these measures as a CSV-file in output folder. 
Parameters:
    path_weighted_edgelist: str <path-to-weighted_edgelist>
    cutoff_filter: int, numerical variable
    output_path: str <path-to-output-dir>
Usage:
    network.py --path_weighted_edgelist <path-to-weighted-edgelist> --cutoff_filter int --output_path <path-to-output-dir>
Example:
    $ python network.py --path_weighted_edgelist weighted_edgelist.csv --cutoff_filter 500 --output_path output/
Output:
    network.png: Network visualization saved in viz folder
    centrality_measures.csv: centrality measures for each node saved in output folder
"""

# Import dependencies #
# System tools
import os
# Data analysis
import pandas as pd
from collections import Counter
from itertools import combinations 
from tqdm import tqdm
# NLP
import spacy
nlp = spacy.load("en_core_web_sm")
# Network visualization
import networkx as nx
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,20)
# Argparse
import argparse

# Define main function
def main():
    
    # Initialise ArgumentParser class
    ap = argparse.ArgumentParser()
    # Argument 1: the path to the edgelist
    ap.add_argument("-i", "--input_weighted_edgelist", required = True, help = "Path to the weighted edgelist saved as a CSV-file")
    # Argument 2: cut-off point to filter data
    ap.add_argument("-c", "--cutoff_filter", required = True, help = "Define the edge weight cut-off point to filter data")
    # Argument 3: the path to the output directory
    ap.add_argument("-o", "--output_path", required = True, help = "Path to output directory")
    # Parse arguments
    args = vars(ap.parse_args())
    
    # Define cut-off value 
    cut_off = args["cutoff_filter"]
    
    # Define path to edgelist
    weighted_edgelist_path = args["input_weighted_edgelist"]
    # Read csv
    weighted_edgelist = pd.read_csv(weighted_edgelist_path)
    # Create dataframe
    edges_df = pd.DataFrame(weighted_edgelist, columns = ["nodeA", "nodeB", "weight"])
    # Filter data based on cut-off point speficied by user
    filtered_edges_df = weighted_edgelist[weighted_edgelist["weight"] > int(cut_off)]

    # Define output path
    output_path = args["output_path"]
    # Create output directory if it doesn't exist already
    if not os.path.exists("output"):
        os.mkdir("output")
        
    # Create viz folder to save the network visualization in
    if not os.path.exists("viz"):
        os.mkdir("viz")
    
    ## NETWORK VISUALIZATION ##
    
    # Crate graph object (G) with networkx
    G = nx.from_pandas_edgelist(filtered_edges_df, "nodeA", "nodeB", ["weight"])
    
    # Plot graph object (G) using pygraphviz
    position = nx.nx_agraph.graphviz_layout(G, prog = "neato")
    
    # Draw the graph
    nx.draw(G, position, with_labels = True, node_size = 20, font_size = 10)
    
    # Save network visualization in viz folder
    output_viz = os.path.join("viz", "network.png")
    plt.savefig(output_viz, dpi = 300, bbox_inches = "tight")
    
    ## CALCUALTE CENTRALITY MEASURES ##
    
    # Calculate degree centrality
    degree_centrality = nx.degree_centrality(G)
    
    # Calculate betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(G)
    
    # Calculate eigenvector centrality
    eigenvector_centrality = nx.eigenvector_centrality(G)
    
    # Create dataframes
    degree_df = pd.DataFrame(degree_centrality.items(), columns = ["node", "degree_centrality"])
    
    betweenness_df = pd.DataFrame(betweenness_centrality.items(), columns = ["node", "betweenness_centrality"])
    
    eigenvector_df = pd.DataFrame(eigenvector_centrality.items(), columns = ["node", "eigenvector_centrality"])
    
    # Merge dataframes
    x = degree_df.merge(betweenness_df)
    centrality_df = x.merge(eigenvector_df).sort_values("degree_centrality", ascending = False)
    
    # Save dataframe as CSV-file in output folder
    output_file = os.path.join(output_path, "centrality_measures.csv")
    centrality_df.to_csv(output_file, index = False)
    
    # Print message to user
    print("Done! Your network visualization has now been saved in the viz folder as 'network.png' and a CSV-file containing measures of centrality for each node has been saved in the output folder as 'centrality_measures.csv'")
    
# Define behaviour when called from command line
if __name__=="__main__":
    main()