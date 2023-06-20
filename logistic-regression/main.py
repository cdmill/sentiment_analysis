"""
A simple implementation of logistic regression for sentiment analysis.
Given an inputted 'tweet', a positve or negative sentiment is outputted
"""

from preprocessing import build_freqs
from logistic_regression import predict_tweet
from sys import argv
import numpy as np
import nltk
from nltk.corpus import twitter_samples


# pretrained logistic regression model
theta = [6.03518871e-08, 5.38184972e-04, -5.58300168e-04]

nltk.download("twitter_samples", quiet=True)
nltk.download("stopwords", quiet=True)

positive_tweets = twitter_samples.strings("positive_tweets.json")
negative_tweets = twitter_samples.strings("negative_tweets.json")

# test_pos = positive_tweets[4000:]
train_pos = positive_tweets[:4000]
# test_neg = negative_tweets[4000:]
train_neg = negative_tweets[:4000]

train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)

train_x = train_pos + train_neg

freqs = build_freqs(train_x, train_y)

""" predict user inputted tweet """

if len(argv) != 2:
    print("Incorrect number of arguments passed")
    exit()

tweet = argv[1]

y_hat = predict_tweet(tweet, freqs, theta)

if y_hat > 0.5:
    print("Positive sentiment")
else:
    print("Negative sentiment")
