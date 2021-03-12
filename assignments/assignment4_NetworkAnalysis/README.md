# Assignment 4: Network Analysis 

### Description of task: Creating reusable network analysis pipeline

The purpose of this asssignment is to create a reusable network analysis pipeline. The script should be able to be run from the command line. It should take any weighted edgelist as an input, providing that edgelist is saved as a CSV with the column headers "nodeA", "nodeB", and "weight". For any given weighted edgelist, the script should be used to create a network visualization, which will be saved in a folder called viz. It should also create a data frame showing the degree, betweenness, and eigenvector centrality for each node. It should save this as a CSV in a folder called output. <br>

### Running the script <br>
Step-by-step-guide:

1. Clone the repository
```
git clone https://github.com/sofieditmer/cds-language.git cds-language-sd
```

2. Navigate to the newly created directory
```
cd cds-language-sd/assignments/assignment4_NetworkAnalysis
```

3. Create and activate virtual environment, "assignment4_venv", by running the bash script create_assignment4_venv.sh

```
bash create_assignment4_venv.sh
source assignment4_venv/bin/activate
```

4. Run the network.py script within the assignment4_venv virtual environment. I have provided a weighted edgelist called "weighted_edgelist.csv" in the data folder on which you can run the script. Run the script and specify the parameters:

`-i:` path to the weighted edgelist <br>
`-c:` edge weight cut-off point to filter the data based on <br>
`-o:` path to output directory

Example: <br>
```
python3 network.py -i ../weighted_edgelist.csv -c 300 -o output/
```

### Output <br>
When running the network.py script you will get two outputs: a network visualization called "network.png" which is saved in a folder called "viz" which is created when running the script, and a CSV-file containing centrality measures for each node called "centrality_measures.csv" saved in the specified output directory. If an output directory is not specified, a directory called "output" will be created in which the CSV-file will be saved.
1. network.png | the network visualization.
2. centrality_measures.csv | centrality measures (degree, betweenness, and eigenvector) for each node.
