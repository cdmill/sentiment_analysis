# README

This is a collection of files used for performing sentiment analysis using methods such as linear regression and naive bayes..

## Logistic Regression

An example implementatin of a logistic regression model that is trained on the NLTK module's twitter corpus.
The model predicts the sentiment of a tweet or inputted text.

If you want to test your own tweet on the model, pass in a string via the command line to the `main.py` file and see what the model predicts.

Example input/output:

```
python3 main.py "today is a good day"
Positive sentiment
```

```
python3 main.py "today is a bad day"
Negative sentiment
```
