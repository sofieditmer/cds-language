# Assignment 3: Dictionary-Based Sentiment Analysis with Python

## Data
For this assignment the following CSV-file, containing 1 million news headlines, was used: <br>
https://www.kaggle.com/therohk/million-headlines

## Running the script
To run the script `sentiment.py` script provided in this repository, it is recommended to run the `create_venv.sh` script first. This creates a virtual environment and installs the required dependencies from the requirements.txt file. 
The outputs of the script will be saved in the output directory specified when runnning the script. 

Step-by-step guide:

1. Clone the repository <br>
`git clone https://github.com/sofieditmer/LanguageAnalytics.git cds-language-sd`

2. Navigate to correct folder in the newly created repository <br>
`cd cds-language-sd/Assignment_3`

3. Create a virtual environment by called "sentiment_environment" running the create_assignment3_venv.sh script <br>
`bash create_assignment3_venv.sh`

4. Activate the virtual environment <br>
`source sentiment_environment/bin/activate`

5. Run the sentiment.py script within the virtual environment and specify the relevant parameters <br>
Example: `python3 sentiment.py -p data/abcnews-date-text.csv -o plots/` <br>
-p is the input path <br>
-o is the output path
