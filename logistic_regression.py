""" implementation of logistic regression for sentiment analysis on tweets from
the nltk corpus
"""

import nltk
from os import getcwd
import numpy as np
import pandas as pd
from nltk.corpus import twitter_samples 
from utils import process_tweet, build_freqs

nltk.download('twitter_samples')
nltk.download('stopwords')
filePath = f"{getcwd()}/../tmp2/"
nltk.data.path.append(filePath)

def sigmoid(z): 
    h = None
    exp = np.exp(-z)
    h = 1/(1+exp)
    return h

def gradientDescent(x, y, theta, alpha, num_iters):
    m, _ = x.shape
    
    for i in range(0, num_iters):
        # get z, the dot product of x and theta
        z = np.dot(x, theta)
        # get the sigmoid of z
        h = sigmoid(z)
        # calculate the cost function
        J = (float(-1)/m) * (np.dot(np.transpose(y),np.log(h))+np.dot(np.transpose(1-y),np.log(1-h)))
        # update the weights theta
        theta = theta - alpha/m * (np.dot(np.transpose(x),(h-y)))

    J = float(J)
    return J, theta

def extract_features(tweet, freqs, process_tweet=process_tweet):
    word_l = process_tweet(tweet)
    
    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3)) 
    
    # bias term is set to 1
    x[0,0] = 1 
    
    # loop through each word in the list of words
    for word in word_l:
        # increment the word count for the positive label 1
        x[0,1] += freqs.get((word,1.0), 0)
        # increment the word count for the negative label 0
        x[0,2] += freqs.get((word,0.0), 0)
        
    assert(x.shape == (1, 3))
    return x

def predict_tweet(tweet, freqs, theta):
    x = extract_features(tweet, freqs)
    y_pred = sigmoid(np.dot(x, theta))
    return y_pred

def test_logistic_regression(test_x, test_y, freqs, theta, predict_tweet=predict_tweet):
    y_hat = []
    
    for tweet in test_x:
        y_pred = predict_tweet(tweet, freqs, theta)
        
        if y_pred > 0.5:
            y_hat.append(1.0)
        else:
            y_hat.append(0.0)

    y_hat_list = np.asarray(y_hat)
    test_y_list = np.squeeze(test_y)
    
    sum = 0
    for i in range(0, len(y_hat)):
        if y_hat_list[i] == test_y_list[i]:
            sum += 1
            
    m, _ = test_y.shape
    accuracy = np.float64(sum/m)
    return accuracy

###

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

test_pos = positive_tweets[4000:]
test_neg = negative_tweets[4000:]
train_pos = positive_tweets[:4000]
train_neg = negative_tweets[:4000]

train_x = train_pos + train_neg 
test_x = test_pos + test_neg

train_y = np.append(np.ones((len(train_pos), 1)),\
                    np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)),\
                   np.zeros((len(test_neg), 1)), axis=0)

freqs = build_freqs(train_x, train_y)

# collect the features 'x' and stack them into a matrix 'X'
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :]= extract_features(train_x[i], freqs)

# training labels corresponding to X
Y = train_y

# Apply gradient descent
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)

