# Assignment 5: Unsupervised Machine Learning

### Description of task: Applying unsupervised machine learning to text data

For this assignment, I have chosen to perform topic modeling and train an LDA model on all tweets written by Donald Trump between May 2009 and June 2020. The dataset I used can be found on Kaggle (link to dataset: https://www.kaggle.com/austinreese/trump-tweets). 

__Research statement__ <br>
I chose to work with the tweets of Donald Trump between 2009 and 2020, because I was interested in whether it would be possible to find meaningful and consistent topics within them. Moreover, I thought it would be interesting to see how the contents of the tweets change over time. 

__Results__ <br>
The LDA model trained on the Donald Trump tweets achieved a perplexity score of -8.0 and a coherence measure of 0.40 with the most optimal number of topics being 10. The outputs of the script are as follows:

1. A txt-file containing an overview of the topics as well as the perplexity and coherence scores. 
2. A csv-file showing the most dominant topic per chunk. 
3. A csv-file showing the topics and their assoicated percentage contribution, i.e. how frequently each topic occurs across the corpus.
4. A plot showing the coherence value as a function of the number of topics. 
5. A lineplot showing showing the rolling mean representing how the content of the tweets change over time in terms of the dominance of the topics. 

All results can be found in the out and the viz directories. 

__Optimal number of topics__ <br>
Below you can see that the optimal number of topics proved to be 10 based on the maximum coherence value.
![alt text](https://github.com/sofieditmer/cds-language/blob/main/assignments/assignment5_TopicModeling/viz/n_topics_coherence.jpg) <br>

__Overview of topics__ <br>
Below is an overview of the 10 topics found by the LDA model. Each topic contains a set of keywords as well as the weight of each individual keyword indicating the importance of that particular keyword for the particular topic.
![alt text](https://github.com/sofieditmer/cds-language/blob/main/assignments/assignment5_TopicModeling/out/overview_topics.png) <br>

__Topics over time__ <br>
Below is an overview of how the contents of the tweets change over time. The dominance of the individual topics change over time, indicating that the tweets change in terms of content from 2009 to 2020. 
![alt text](https://github.com/sofieditmer/cds-language/blob/main/assignments/assignment5_TopicModeling/viz/topics_over_time.jpg) <br>

__Conclusion__ <br>
Overall, meaningful and consitent topics were found, however, more preprocessing would prove to be beneficial for the clarity of the topics. What is interesting to see is that topic 2 which contains keywords such as "people", "country", "border", "job", and "deal" is very prominent in the beginning and in the end while topic 0 and 3 containing keywords such as "status", "thank", "trump", "president", "show", "business" are more prominent in between. Further analyses would be interesting to pursue. 

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