# Assignment 5: Unsupervised Machine Learning

### Description of task: Applying unsupervised machine learning to text data

For this assignment, I have chosen to perform topic modeling and train an LDA model on all tweets written by Donald Trump between May 2009 and June 2020. The dataset I used can be found on Kaggle (link to dataset: https://www.kaggle.com/austinreese/trump-tweets). 

__Research statement__
I chose to work with the tweets of Donald Trump between 2009 and 2020, because I was interested in whether it would be possible to find meaningful and consistent topics within them.

__Results__
The LDA model trained on the Donald Trump tweets achieved a perplexity score of -8.43 and a coherence measure of 0.37. The outputs of the script are as follows:

1. A csv-file showing the most dominant topic per chunk. 
2. A csv-file showing the topics and their assoicated percentage contribution, i.e. how frequently each topic occurs across the corpus. 
3. A document-topic matrix showing have all chunks for each topic
4. A lineplot showing the rolling scores calculated with a rolling mean of 50 as default. This lineplot demonstrates how chunks of tweets change in content over time, i.e. which topic is most dominant over time. 

All results can be found in the out and viz directories. 

__Overview of topics__
![alt text](https://github.com/sofieditmer/cds-language/blob/main/assignments/assignment5_TopicModeling/out/topics.png)

__Conclusion__
Some topics are more clear than others. If I had had more time, I would look into that. I would also like to look at how the contents of the tweets change over time, i.e. what Donald Trump is generally concerned with over the years. 

<br>


### Running the script <br>
Step-by-step-guide:

1. Clone the repository
```
git clone https://github.com/sofieditmer/cds-language.git cds-language-sd
```

2. Navigate to the newly created directory
```
cd cds-language-sd/assignments/assignment5_TopicModeling
```

3. Create and activate virtual environment, "ass5_venv", by running the bash script ass5_venv.sh. This will install the required dependencies listed in requirements.txt 

```
bash create_ass5_venv.sh
source ass5_venv/bin/activate
```

4. Navigate to the src folder

```
cd src
```

4. Run the lda.py script within the ass5_venv virtual environment.

```
$ python lda.py
```