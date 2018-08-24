#Step 1: import packages
import pandas as pd
import numpy as np
from sklearn import svm

#Step 2: Load the data
df = pd.read_excel('modified dataset for sentiment analysis.xlsx',header=0, delimiter="\t", quoting=3)
df.dropna(inplace=True)
print df.shape
print df.head(10)

#Step 3: split the loaded data has a train data and test data
from sklearn.model_selection import train_test_split
df["TARGET"]=np.where(df['target']>0,1,-1)
X_train, X_test, Y_train, Y_test = train_test_split(df['Feedback'],df['target'],random_state=0)

#Step 4: count repeated words in the training set
from collections import Counter

count_vocab = Counter()
for txt in X_train:
    for word in txt.split(' '):
        count_vocab[word] += 1
        
print count_vocab.most_common(10)

#Step 5: stopwords

# A stop word is a commonly used word (such as “the”, “a”, “an”, “in”)that a search engine has been programmed to ignore.

# Our Input here:
#['This', 'is', 'a', 'sample', 'sentence', ',', 'showing', 'off', 'the', 'stop', 'words', 'filtration', '.']

# Our output here:
#['This', 'sample', 'sentence', ',', 'showing', 'stop', 'words', 'filtration', '.']

import nltk
nltk.download('stopwords')


from nltk.corpus import stopwords
stop = stopwords.words('english')

vocab_reduced = Counter()
for w, c in count_vocab.items():
    if not w in stop:
        vocab_reduced[w] = c
print vocab_reduced.most_common(14)

#Step 6: re - regular expression

# A regular expression (or RE) specifies a set of strings that matches it.
# .....the functions in this module let you check if a particular string matches a given regular expression (or if a given regular expression matches a particular string.
# ..........which comes down to the same thing).
import re

def preprocessor(text):
    
    #remove HTML markup
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    # Remove any non-word character and append the emoticons,
    # removing the nose character for standarization. Convert to lower case
    text = (re.sub('[\W]+', ' ', text.lower()) + ' ' + ' '.join(emoticons).replace('-', ''))
    return text

print(preprocessor('This!! twit man :) is <b>nice</b>'))


# Step 7: PorterStemmer

# The idea of stemming is a sort of normalizing method. Many variations of words carry the same meaning, other than when tense is involved.
# The reason why we stem is to shorten the lookup, and normalize sentences.

# Example 1:

# I was taking a ride in the car.
# I was riding in the car.

# Example 2:

# loving - love
# love - loving

from nltk.stem import PorterStemmer
porter = PorterStemmer()

def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

print(tokenizer('Hi there, I am loving this, like with a lot of love'))
print(tokenizer_porter('Hi there, I am loving this, like with a lot of love'))


# Step 8: Pipelining of models

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None)
param_grid = [{'vect__ngram_range': [(1, 9)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__preprocessor': [None, preprocessor],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 9)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__preprocessor': [None, preprocessor],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__C': [1.0, 10.0, 100.0]},
              ]
lr_tfidf = Pipeline([('vect', tfidf),('clf', LinearSVC(random_state=0))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,scoring='accuracy',cv = 5,verbose=1,n_jobs=-1)

# Step 9: training our model with dataset
# It takes some time 

gs_lr_tfidf.fit(X_train, Y_train)

# Step 10: Finding a Best papameter and Best accuracy of trained model.  

print('Best parameter set: ' + str(gs_lr_tfidf.best_params_))
print('Best accuracy: %.3f' % gs_lr_tfidf.best_score_)

# Step 11: Accuracy of model prediction.

clf = gs_lr_tfidf.best_estimator_
print('Accuracy in test: %.3f' % clf.score(X_test, Y_test))

# pickle file
import pickle
import os
fname="model.sav"

pickle.dump(clf, open(fname, 'wb'), protocol=2)

# Step 12: manual testing

twits = []
user_feedback = raw_input("give ur feedback: ")
twits.append(user_feedback)

preds = clf.predict(twits)
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