#Natural Language Processing

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#Cleaning the texts
import re
import nltk  #Library for NLP
nltk.download('stopwords')  #downloads irrelevant words like 'this', 'that' etc
from nltk.corpus import stopwords  # import the stopwords
from nltk.stem.porter import PorterStemmer # to keep the root of each word so that only one version of each word remains in the sparse matrix
corpus = []
for i in range(0,1000):
    
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i] ) #remove all characters except letters
    review = review.lower()   #change the words to lower cases
    review = review.split()   # split the list of words to separate words
    ps = PorterStemmer() #new object for stemming
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] # removes words from review it is not present in stopwords
    review = ' '.join(review) #joining the different words separated by space
    corpus.append(review) #store all of the cleaned reviews in corpus
    
#Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

                
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
