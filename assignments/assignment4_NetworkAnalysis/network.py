#!/usr/bin/env python

"""
For any given weighted edgelist of named entities based on document co-occurence, create a network visualization and save that in viz folder, create dataframe showing degree, betweenness, and eigenvector centrality for each node, and save these measures as a CSV-file in output folder. 

Parameters:
    weighted_edgelist: str <path-to-weighted-edgelist>
    cutoff_edgeweight: int <minimum-edgeweight>

Usage:
    network.py --weighted_edgelist <path-to-weighted-edgelist> --cutoff_edgeweight <minimum-edgeweight>

Example:
    $ python network.py --weighted_edgelist ../data/weighted_edgelist.csv --cutoff_edgeweight 500

Output:
    network.png: Network visualization saved in viz folder
    centrality_measures.csv: centrality measures for each node saved in output folder
"""

### DEPENDENCIES ###

import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,20)
import argparse

### MAIN FUNCTION ###

def main():
    
    # Initialise ArgumentParser class
    ap = argparse.ArgumentParser()
    
    # Argument 1: the path to the edgelist
    ap.add_argument("-i", "--input_weighted_edgelist", required = True, help = "Path to the weighted edgelist saved as a CSV-file")
    
    # Argument 2: cut-off edgeweight to filter data based on
    ap.add_argument("-c", "--cutoff_edgeweight", required = True, help = "Define the edge weight cut-off point to filter data")
    
    # Parse arguments
    args = vars(ap.parse_args())
    
    # Define path to edgelist
    weighted_edgelist = args["input_weighted_edgelist"]
    
    # Define cut-off value 
    cutoff_edgeweight = args["cutoff_edgeweight"]
    
    # Create output directories: output and viz
    if not os.path.exists("output"):
        os.mkdir("output")
        
    if not os.path.exists("viz"):
        os.mkdir("viz")
        
    # Run the network analysis
    Network_analysis(weighted_edgelist, cutoff_edgeweight)
        
### NETWORK ANALYSIS ### 

class Network_analysis:
    """ 
    Network Analysis Class
    Input: 
        weighted edgelist: str <path-to-weighted-edgelist>
        cutoff_edgeweight: int <minimum-edgeweight>
    """
    
    # Intialize Network_analysis class
    def __init__(self, weighted_edgelist, cutoff_edgeweight):
        
        # Load data
        weighted_edgelist_df = pd.read_csv(weighted_edgelist)
        
        # Create network visualization
        network = self.network(weighted_edgelist_df, cutoff_edgeweight)
        
        # Calculate centrality measures
        centrality_measures = self.centrality_measures(network)
        
        # User message
        print("Done! Your network visualization has now been saved in the viz folder as 'network.png' and a CSV-file containing measures of centrality for each node has been saved in the output folder as 'centrality_measures.csv'")
    
    # Network visualization function 
    def network(self, weighted_edgelist_df, cut_off):
        """ 
        Network function that creates a network graph and saves it as a .png file. 
        Input: 
            weighted edgelist: str <path-to-weighted-edgelist>
            cutoff_edgeweight: int <minimum-edgeweight>
        """
        
        # Filter data based on cut-off edgeweight speficied by user
        filtered_edges_df = weighted_edgelist_df[weighted_edgelist_df["weight"] > int(cut_off)]
        
        # Create graph using the networkx library
        network_graph = nx.from_pandas_edgelist(filtered_edges_df, "nodeA", "nodeB", ["weight"])
        
        # Plot graph object (G) using pygraphviz
        position = nx.nx_agraph.graphviz_layout(network_graph, prog = "neato")
        
        # Draw the graph
        nx.draw(network_graph, position, with_labels = True, node_size = 20, font_size = 10)
        
        # Save network graph in viz folder
        output_viz = os.path.join("viz", "network.png")
        plt.savefig(output_viz, dpi = 300, bbox_inches = "tight")

        return network_graph
    
    # Centrality measures calculation function
    def centrality_measures(self, network_graph):
        """ 
        Centrality measures calculation function that calculates degree centrality, betweenness centrality, and eigenvector centrality for each edgea and saves the results as a CSV-file in the output directory. 
        Input: 
            network_graph
        """
        
        # Calculate degree centrality
        degree_centrality = nx.degree_centrality(network_graph)
        # Create dataframe for degree centrality
        degree_df = pd.DataFrame(degree_centrality.items(), columns = ["node", "degree_centrality"])
        
        # Calculate betweenness centrality
        betweenness_centrality = nx.betweenness_centrality(network_graph)
        # Create dataframe for betweenness centrality
        betweenness_df = pd.DataFrame(betweenness_centrality.items(), columns = ["node", "betweenness_centrality"])
        
        # Calculate eigenvector centrality
        eigenvector_centrality = nx.eigenvector_centrality(network_graph)
        # Create dataframe for eigenvector centrality
        eigenvector_df = pd.DataFrame(eigenvector_centrality.items(), columns = ["node", "eigenvector_centrality"])
        
        # Merge dataframes into one dataframe containing the measures for each node
        centrality_df = degree_df.merge(betweenness_df)
        centrality_df = centrality_df.merge(eigenvector_df).sort_values("degree_centrality",
                                                            ascending = False)
        
        # Save dataframe as CSV-file
        output_csv = os.path.join("output", "centrality_measures.csv")
        centrality_df.to_csv(output_csv, index = False)
    
    
# Define behaviour when called from command line
if __name__=="__main__":
    main()