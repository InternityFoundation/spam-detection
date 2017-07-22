## Importing required packages
import os
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.metrics import confusion_matrix

## Creating a word dictionary
def make_Dictionary(train_dir):
    
    email_files = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]
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
    
    dictionary = dictionary.most_common(3000)
    
    return dictionary

## Feature Extraction Process
def extract_features(train_dir): 
    
    email_files = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]
    features_matrix = np.zeros((len(email_files),3000))
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

## Training our classifier using Naive Bayes and SVM
# Create a dictionary of words with its frequency
train_dir = 'train-mails'
dictionary = make_Dictionary(train_dir)

# Prepare feature vectors per training mail and its labels
train_labels = np.zeros(702)
train_labels[351:701] = 1
train_matrix = extract_features(train_dir)

# Training SVM and Naive bayes classifier
model1 = MultinomialNB(alpha=0.5)
model2 = LinearSVC()

model1.fit(train_matrix,train_labels)
model2.fit(train_matrix,train_labels)

## Predicting test dataset using our classifiers
# Test the unseen mails for Spam
test_dir = 'test-mails'
test_matrix = extract_features(test_dir)
test_labels = np.zeros(260)
test_labels[130:260] = 1
result1 = model1.predict(test_matrix)
result2 = model2.predict(test_matrix)

## Finding out accuracy of our models
print confusion_matrix(test_labels,result1)
print confusion_matrix(test_labels,result2)