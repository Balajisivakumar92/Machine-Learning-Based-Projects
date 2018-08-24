import os
import pickle
import re

#from flask import Flask, request, jsonify

# Unpickle the trained classifier and write preprocessor method used
def tokenizer(text):
    return text.split(' ')

def preprocessor(text):
    """ Return a cleaned version of text
    """
    # Remove HTML markup
    text = re.sub('<[^>]*>', '', text)
    # Save emoticons for later appending
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    # Remove any non-word character and append the emoticons,
    # removing the nose character for standarization. Convert to lower case
    text = (re.sub('[\W]+', ' ', text.lower()) + ' ' + ' '.join(emoticons).replace('-', ''))

    return text

fname="model.sav"
loaded=pickle.load(open(fname, 'rb'))
twits = []
user_feedback = raw_input("give ur feedback: ")
twits.append(user_feedback)
#tweet_classifier = pickle.load(open('../data/logisticRegression.pkl', 'rb'))

preds = loaded.predict(twits)

print preds

for i in range(len(twits)):
    #print(preds[i], twits[i])
    if preds[i] == -1:
        print "Feedback Negative: {BAD}= ",(preds[i])
    elif preds[i] == 1:
        print "Feedback Positive: {GOOD}=",(preds[i])
    else:
        print "Feedback neutral: {AVERAGE}=",(preds[i])

print('DONE :)')