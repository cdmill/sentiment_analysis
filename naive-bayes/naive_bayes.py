"""
A collection of methods used to perform sentiment analysis via Naive Bayes.
"""

import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


def process_tweet(tweet):
    """
    Input:
        tweet: a string containing a tweet
    Output:
        cleaned_tweet: the processed tweet as a list of words
    """

    stemmer = PorterStemmer()
    stopwords_english = stopwords.words("english")
    # remove stock market tickers like $GE
    tweet = re.sub(r"\$\w*", "", tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r"^RT[\s]+", "", tweet)
    # remove hyperlinks
    tweet = re.sub(r"https?://[^\s\n\r]+", "", tweet)
    # remove hashtags
    tweet = re.sub(r"#", "", tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    cleaned_tweet = []
    for word in tweet_tokens:
        if (
            word not in stopwords_english
            and word not in string.punctuation  # remove stopwords
        ):  # remove punctuation
            stem_word = stemmer.stem(word)  # stemming word
            cleaned_tweet.append(stem_word)

    return cleaned_tweet


def count_tweets(result, tweets, sentiments):
    """
    Input:
        result: a dictionary that will be returned with frequency mappings
        tweets: a list of tweets
        sentiments: a list of sentiment labels {0,1} corresponding to each tweet
    Output:
        result: a dictionary mapping each pair to its frequency
    """

    for sentiment, tweet in zip(sentiments, tweets):
        for word in process_tweet(tweet):
            pair = (word, sentiment)

            if pair in result:
                result[pair] += 1
            else:
                result[pair] = 1
    return result


def train(freqs, train_y):
    """
    Input:
        freqs: a dictionary that maps a (word, label) pair to its frequency
        train_y: a list of sentiment labels {0,1} corresponding to each tweet
    Output:
        logprior: the logprior for the given Input
        loglikelihood: the loglikelihood of the Naive Bayes equation
    """

    loglikelihood = {}
    logprior = 0

    # calculate V := the number of unique terms (types) in the vocabulary
    vocab = [pair[0] for pair in freqs.keys()]
    V = len(set(vocab))

    # calculate the number of positive/negative words
    num_pos = num_neg = 0
    for pair in freqs.keys():
        # if sentiment label is positive
        if pair[1] > 0:
            num_pos += freqs[pair]
        else:
            num_neg += freqs[pair]

    D_pos = D_neg = 0

    for sentiment in train_y:
        if sentiment == 1:
            D_pos += 1
        else:
            D_neg += 1

    # calculate logprior
    logprior = np.log(D_pos) - np.log(D_neg)

    for word in vocab:
        # get positive/negative frequency of word
        freq_pos = freqs.get((word, 1), 0)
        freq_neg = freqs.get((word, 0), 0)

        # calculate probability that each word is positive/negative
        prob_pos = (freq_pos + 1) / (num_pos + V)
        prob_neg = (freq_neg + 1) / (num_neg + V)

        loglikelihood[word] = np.log(prob_pos) - np.log(prob_neg)
    return logprior, loglikelihood


def predict(tweet, logprior, loglikelihood):
    """
    Input:
        tweet: a string
        logprior: returned from train()
        loglieklihood: returned from train()
    Output:
        p = sum of the loglikelihoods of each word in the tweet + logprior
    """

    tokens = process_tweet(tweet)

    p = 0
    p += logprior

    for word in tokens:
        if word in loglikelihood:
            p += loglikelihood[word]
    return p
