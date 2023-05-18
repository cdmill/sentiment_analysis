""" Example file showing how to process text and build frequency data for it.
The file uses tweets from the NLTK module's twitter corpus.
"""

import re
import string
import numpy as np

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
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks    
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    # remove hashtags
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    cleaned_tweet = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            stem_word = stemmer.stem(word)  # stemming word
            cleaned_tweet.append(stem_word)

    return cleaned_tweet

def build_freqs(tweets, sentiment_vector):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        sentiment_vector: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its frequency
    """

    sentiment_list = np.squeeze(sentiment_vector).tolist()

    freqs = {}
    for sentiment, tweet in zip(sentiment_list, tweets):
        for word in process_tweet(tweet):
            pair = (word, sentiment)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs

