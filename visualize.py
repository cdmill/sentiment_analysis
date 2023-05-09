""" this file plots the data from the sentiment analysis on tweets from the
nltk corpus using matplotlib
"""


import nltk                         
from os import getcwd
import pandas as pd                 
from nltk.corpus import twitter_samples 
import matplotlib.pyplot as plt    
import numpy as np                  
from utils import process_tweet, build_freqs 

nltk.download('twitter_samples')

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

tweets = positive_tweets + negative_tweets 
labels = np.append(np.ones((len(positive_tweets),1)),\
                   np.zeros((len(negative_tweets),1)), axis = 0)

train_pos  = positive_tweets[:4000]
train_neg  = negative_tweets[:4000]

train_x = train_pos + train_neg
data = pd.read_csv('./data/logistic_features.csv')

X = data[['bias', 'positive', 'negative']].values 
Y = data['sentiment'].values

# pretrained logistic regression model
theta = [6.03518871e-08, 5.38184972e-04, -5.58300168e-04]

# Plot the samples using columns 1 and 2 of the matrix
fig, ax = plt.subplots(figsize = (8, 8))

colors = ['red', 'green']

# Color based on the sentiment Y
ax.scatter(X[:,1], X[:,2], c=[colors[int(k)] for k in Y], s = 0.1)  
plt.xlabel("Positive")
plt.ylabel("Negative")

# Equation for the separation plane
def neg(theta, pos):
    return (-theta[0] - pos * theta[1]) / theta[2]

# Equation for the direction of the sentiments change
def direction(theta, pos):
    return    pos * theta[2] / theta[1]

# Plot the samples using columns 1 and 2 of the matrix
fig, ax = plt.subplots(figsize = (8, 8))
colors = ['red', 'green']

# Color base on the sentiment Y
ax.scatter(X[:,1], X[:,2], c=[colors[int(k)] for k in Y], s = 0.1) 
plt.xlabel("Positive")
plt.ylabel("Negative")

maxpos = np.max(X[:,1])
offset = 5000 # The pos value for the direction vectors origin

# Plot a gray line that divides the 2 areas.
ax.plot([0,  maxpos], [neg(theta, 0),   neg(theta, maxpos)], color = 'gray') 

# Plot a green line pointing to the positive direction
ax.arrow(offset, neg(theta, offset), offset, direction(theta, offset),\
         head_width=500, head_length=500, fc='g', ec='g')
# Plot a red line pointing to the negative direction
ax.arrow(offset, neg(theta, offset), -offset, -direction(theta, offset),\
         head_width=500, head_length=500, fc='r', ec='r')

plt.show()

