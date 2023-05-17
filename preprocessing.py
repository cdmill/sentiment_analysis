""" Example file showing how to preprocess text. This file makes use of the
twitter corpus from the nltk module
"""

import re
import string
import nltk
import random
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

nltk.download("twitter_samples")
nltk.download("stopwords")

positive_tweets = twitter_samples.strings("positive_tweets.json")
negative_tweets = twitter_samples.strings("negative_tweets.json")

size = len(positive_tweets)
tweet = positive_tweets[random.randrange(size)]
print("original:", tweet, sep='\n')
print('\n')

# remove hyperlinks, mentions, and hashtags from tweets
tweet_modified = re.sub(r'^RT[\s]+', '', tweet)
tweet_modified = re.sub(r'https?://[^\s\n\r]+', '', tweet_modified)
tweet_modified = re.sub(r'#', '', tweet_modified)
print("hyperlinks, mentions, and hashtags removed:", tweet_modified, sep='\n')
print('\n')

# tokenize
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
tweet_tokens = tokenizer.tokenize(tweet_modified)
print("tokenized:", tweet_tokens, sep='\n')
print('\n')

# remove stop words from tweets
stopwords_english = stopwords.words("english")
tweets_clean = []
for word in tweet_tokens:
    if word not in stopwords_english and word not in string.punctuation:
        tweets_clean.append(word)

print("stop words removed:", tweets_clean, sep='\n')
print('\n')

# stem the tweets
stemmer = PorterStemmer()
tweets_stem = []
for word in tweets_clean:
    stem_word = stemmer.stem(word)
    tweets_stem.append(stem_word)

print("stemmed:", tweets_stem, sep='\n')
print('\n')

