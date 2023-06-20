# README

This is a collection of files used for performing sentiment analysis using methods such as linear regression and naive bayes.

Each sentiment analysis model is trained on the NLTK module's twitter corpus.
The models predicts the sentiment of a tweet or inputted text.

If you want to test your own tweet on the model, pass in a string via the command line to the `main.py` file and see what the model predicts.

## Logistic Regression

Example input/output:

```
python3 main.py "today is a good day"
Positive sentiment
```

```
python3 main.py "today is a bad day"
Negative sentiment
```

## Naive bayes

Example input/output:

```
python3 main.py "today is a good day"
1.3093132593867
```

```
python3 main.py "today is a bad day"
-0.8166482979280296
```
