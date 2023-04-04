""" Example file showing how to preprocess text. This file makes use of the
twitter corpus from the nltk module
"""

import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import random
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

nltk.download("twitter_samples")
nltk.download("stopwords")
all_positive_tweets = twitter_samples.strings("positive_tweets.json")
all_negative_tweets = twitter_samples.strings("negative_tweets.json")

fig = plt.figure(figsize=(5, 5))
labels = "Positives", "Negatives"
sizes = [len(all_positive_tweets), len(all_negative_tweets)]
plt.pie(sizes, labels=labels, autopct="%1.1f%%", shadow=True, startangle=90)
plt.axis("equal")
plt.show()

tweet = all_positive_tweets[2277]
tweet_modified = re.sub(r'^RT[\s]+', '', tweet)
tweet_modified = re.sub(r'https?://[^\s\n\r]+', '', tweet_modified)
tweet_modified = re.sub(r'#', '', tweet_modified)

tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
tweet_tokens = tokenizer.tokenize(tweet_modified)

stopwords_english = stopwords.words("english")

tweets_clean = []
for word in tweet_tokens:
    if word not in stopwords_english and word not in string.punctuation:
        tweets_clean.append(word)

stemmer = PorterStemmer()
tweets_stem = []
for word in tweets_clean:
    stem_word = stemmer.stem(word)
    tweets_stem.append(stem_word)

print(tweets_stem)

