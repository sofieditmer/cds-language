# Assignment 3: Dictionary-Based Sentiment Analysis with Python

## Data
For this assignment the following CSV-file, containing 1 million news headlines, was used: <br>
https://www.kaggle.com/therohk/million-headlines

## Running the script
To run the script `sentiment.py`, it is recommended to run the `create_venv.sh` script first. This creates a virtual environment and installs the required dependencies from the requirements.txt file. 
The outputs of the script will be saved in the output directory specified when runnning the script. 

Step-by-step guide:

1. Clone the repository <br>
`git clone https://github.com/sofieditmer/cds-language.git cds-language-sofie`

2. Navigate to the assignment folder in the newly created repository <br>
`cd cds-language-sd/Assignment3_SentimentAnalysis`

3. Create a virtual environment called "sentiment_environment" by running the create_assignment3_venv.sh script <br>
`bash create_assignment3_venv.sh`

4. Activate the newly created virtual environment <br>
`source sentiment_environment/bin/activate`

5. Run the sentiment.py script within the virtual environment and specify the relevant parameters <br>
Example: `python3 sentiment_au617836.py -p ../data/abcnews-date-text.csv -o outputs/` <br>
-p is the input path <br>
-o is the output path

## Summary of results

### Plot 1: Sentiment scores with 1-week rolling average
![Image of Yaktocat](https://github.com/sofieditmer/cds-language/blob/main/assignments/assignment3_SentimentAnalysis/outputs/smoothed_sentiment_week.png)

### Plot 2: Sentiment scores with 1-month rolling average
![Image of Yaktocat](https://github.com/sofieditmer/cds-language/blob/main/assignments/assignment3_SentimentAnalysis/outputs/smoothed_sentiment_month.png)

### General trends and inferences
With both a 1-week and 1-month rolling average the sentiment scores of the news headlines generally lie above 0 indicating a slightly positive sentiment. Hence, it seems that most news headlines from this particular news station display a more positive sentiment as opposed to a negative sentiment. With a 1-week rolling average the sentiment scores generally range between 0 and 0.05 while a less variation is present with a 1-month rolling average with a slight increase in sentiment scores from 2012-2016.