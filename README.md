# README

This is a collection of files used for performing sentiment analysis using linear regression.

## Preprocessing

Before doing anything with text it must be preprocessed.
Because I am using the NLTK module, I will be using the twitter corpus.

In order to preprocess tweets any hyperlinks, hashtags, and mentions need to be removed.
Then, we can tokenize the tweet and remove any stopwords.
Finally, we can stem the tokens.

An example output:

```
original:
  Stats for the day have arrived. 1 new follower and NO unfollowers :) via http://t.co/Smqz6YKvEc.
  
hyperlinks, mentions, and hashtags removed:
  Stats for the day have arrived. 1 new follower and NO unfollowers :) via

tokenized:
  ['stats', 'for', 'the', 'day', 'have', 'arrived', '.', '1', 'new', 'follower', 'and', 'no', 'unfollowers', ':)', 'via']
  
stop words removed:
  ['stats', 'day', 'arrived', '1', 'new', 'follower', 'unfollowers', ':)', 'via']

stemmed:
  ['stat', 'day', 'arriv', '1', 'new', 'follow', 'unfollow', ':)', 'via']
```
