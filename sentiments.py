from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity

from nltk.sentiment.vader import SentimentIntensityAnalyzer
text='''The patient is not suicidal anymore.
The patient is extremely great in health and mentally.
He seems to be demonstrating great skills and improvements.
He doesn't show any improvement in health anymore.
The patient is doing extremely terrible.
The patient's health has worsened a lot'''
from nltk import tokenize
lines_list = tokenize.sent_tokenize(text)
sid = SentimentIntensityAnalyzer()
for sentence in lines_list:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]))
    print("\n")
