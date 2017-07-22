## Importing required packages
import os
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

## Function for creating a word dictionary
def make_Dictionary(train_dir):
    
    email_files = [os.path.join(train_dir,f) for f in sorted(os.listdir(train_dir))]
    stop_words = set(stopwords.words('english'))
    all_words = []       
    
    for mail in email_files:    
        with open(mail) as m:
            for i,line in enumerate(m):
                if i == 2:  #Body of email is only 3rd line of text file
                    list_words = word_tokenize(line)
                    all_words += list_words
                    
    dictionary = Counter(all_words)
    
    list_to_remove = dictionary.keys()

    for item in list_to_remove:
        if item.isalpha() == False: 
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
        elif item in stop_words:
            del dictionary[item]
    
    dictionary = dictionary.most_common(5000)
    
    return dictionary

## Function for the process of Feature Extraction
def extract_features(mail_dir): 
    
    email_files = [os.path.join(mail_dir,f) for f in sorted(os.listdir(mail_dir))]
    features_matrix = np.zeros((len(email_files),5000))
    docID = 0
    
    for mail in email_files:
        with open(mail) as m:
            for i,line in enumerate(m):
                if i == 2:
                    words = word_tokenize(line)
                    
                    for word in words:
                        wordID = 0                      
                        
                        for i,d in enumerate(dictionary):
                            if d[0] == word:
                                wordID = i
                                features_matrix[docID,wordID] = words.count(word)
                                
        docID = docID + 1     
    return features_matrix

## Create a dictionary of words with its frequency
train_dir = 'train-mails'
dictionary = make_Dictionary(train_dir)

## Prepare feature vectors of each training mail and its labels
train_labels = np.zeros(702)
train_labels[351:701] = 1

train_matrix = extract_features(train_dir)

## Training Naive Bayes and SVM classifier
NBmodel = MultinomialNB()
NBmodel.fit(train_matrix,train_labels)

SVMmodel = LinearSVC()
SVMmodel.fit(train_matrix,train_labels)

## Predicting the type of mail of test dataset using our models
test_dir = 'test-mails'

test_labels = np.zeros(260)
test_labels[130:260] = 1

test_matrix = extract_features(test_dir)

NBmodelresult = NBmodel.predict(test_matrix)
SVMmodelresult = SVMmodel.predict(test_matrix)

## Finding out accuracy of our models
print 'Confusion Matrix of our Naive Bayes Classifier is:'
print confusion_matrix(test_labels,NBmodelresult)

print '\nConfusion Matrix of our SVM Classifier is:'
print confusion_matrix(test_labels,SVMmodelresult)
