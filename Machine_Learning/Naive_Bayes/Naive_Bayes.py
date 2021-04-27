#references:
# https://towardsdatascience.com/text-classification-using-naive-bayes-theory-a-working-example-2ef4b7eb7d5a
# https://medium.com/@awantikdas/a-comprehensive-naive-bayes-tutorial-using-scikit-learn-f6b71ae84431
# https://coderzcolumn.com/tutorials/machine-learning/scikit-learn-sklearn-naive-bayes

import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt #https://github.com/matplotlib/matplotlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Load the dataset
data = pd.read_csv('Datasets/test.csv')
# Get the text categories
text_categories = data.info()

#limit the categories to just the 2 that we need
data = data[["publication","content"]]
print(data.publication.value_counts())


#Processing Data
##Tokenization of data (breaking apart words)
data["content"] = data.content.map(lambda x: word_tokenize(x))
##Stemming of data (Converting words to base forms such as likely -> like)
stemmer = PorterStemmer()
data["content"] = data.content.map(lambda l: [stemmer.stem(word) for word in l])
data.content = data.content.str.join(sep=' ')

cv = CountVectorizer(stop_words='english')
data_tf = cv.fit_transform(data.content)
#Creation of training set and testing set
trainX,testX,trainY,testY = train_test_split(data_tf,data.publication)

#Creation of model
mnb = MultinomialNB()
mnb.fit(trainX,trainY)
mnb.class_prior
y_pred = mnb.predict(testX)
mat = confusion_matrix(y_true=testY, y_pred=y_pred,normalize='true')

#Creation of heatmap
sns.heatmap(mat.T, square = True, annot=True, fmt = ".4f", xticklabels=data.publication.unique(),yticklabels=data.publication.unique())
plt.xlabel("true labels")
plt.ylabel("predicted label")
plt.show()

print("The accuracy is {}".format(accuracy_score(testY, y_pred)))
