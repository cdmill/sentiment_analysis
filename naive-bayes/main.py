"""
A simple implementation of Naive Bayes for sentiment analysis.
Given an inputted 'tweet', its sentiment score is outputted.
"""

from naive_bayes import *
from sys import argv
import numpy as np
import nltk
from nltk.corpus import twitter_samples


nltk.download("twitter_samples", quiet=True)
nltk.download("stopwords", quiet=True)

positive_tweets = twitter_samples.strings("positive_tweets.json")
negative_tweets = twitter_samples.strings("negative_tweets.json")

train_pos = positive_tweets[:4000]
train_neg = negative_tweets[:4000]
train_x = train_pos + train_neg

train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)))

freqs = count_tweets({}, train_x, train_y)
logprior, loglikelihood = train(freqs, train_y)

if len(argv) != 2:
    print("Incorrect number of arguments passed")
    exit()

tweet = argv[1]

p = predict(tweet, logprior, loglikelihood)
print("The sentiment of the inputted tweet is", p)
