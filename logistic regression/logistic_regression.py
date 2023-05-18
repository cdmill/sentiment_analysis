""" Implementation of logistic regression for sentiment analysis on tweets from
the NLTK module's twitter corpus
"""

import nltk
from os import getcwd
import numpy as np
from preprocessing import process_tweet

nltk.download('twitter_samples')
nltk.download('stopwords')
filePath = f"{getcwd()}/../tmp2/"
nltk.data.path.append(filePath)

def sigmoid(z): 
    """
    Input:
        z: scalar or array 
    Output:
        h: the sigmoid of z
    """

    h = None
    exp = np.exp(-z)
    h = 1/(1+exp)
    return h

def gradientDescent(x, y, theta, alpha, num_iters):
    '''
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations to train the model
    Output:
        J: the final cost
        theta: the final weight vector
    '''

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
    '''
    Input: 
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output: 
        x: a feature vector of dimension (1,3)
    '''

    word_l = process_tweet(tweet)
    
    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3)) 
    
    # bias term is set to 1
    x[0,0] = 1 
    
    for word in word_l:
        # increment the word count for the positive label 1
        x[0,1] += freqs.get((word,1.0), 0)
        # increment the word count for the negative label 0
        x[0,2] += freqs.get((word,0.0), 0)
        
    assert(x.shape == (1, 3))
    return x

def predict_tweet(tweet, freqs, theta):
    '''
    Input: 
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output: 
        y_pred: the probability of a tweet being positive or negative
    '''

    x = extract_features(tweet, freqs)
    y_pred = sigmoid(np.dot(x, theta))
    return y_pred

def test_logistic_regression(test_x, test_y, freqs, theta, predict_tweet=predict_tweet):
    """
    Input: 
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output: 
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """

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

