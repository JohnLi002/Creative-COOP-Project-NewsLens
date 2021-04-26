import tensorflow_hub as hub
import tensorflow as tf
import pandas as pd
import keras
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.layers import Input, Lambda, Dense
from keras.models import Model
from keras.utils import to_categorical
import seaborn as sns
import matplotlib.pyplot as plt #https://github.com/matplotlib/matplotlib
from sklearn.metrics import confusion_matrix, accuracy_score

def encode(le, labels):
    enc = le.transform(labels)
    return to_categorical(enc)

def decode(le, one_hot):
    dec = np.argmax(one_hot, axis = 1)
    return le.inverse_transform(dec)

def UniversalEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)))

def HeatMap(real, predicted, le):
    #Creation of heatmap
    y_unique = np.array(Y)
    y_unique = np.unique(y_unique) #unique Y values
    decoded_predicted =decode(le, predicted)
    mat = confusion_matrix(y_true=real, y_pred=decoded_predicted)
    sns.heatmap(mat.T, square = True, annot=True, fmt = "d", xticklabels=y_unique,yticklabels=y_unique,normalize=normalize)
    plt.xlabel("true labels")
    plt.ylabel("predicted label")
    plt.title("Predicted Values vs. Real Values")
    plt.figure(figsize=(10, 16)) #change size
    plt.show()
    #accuracy report based on given confusion matrix
    print("The accuracy is {}".format(accuracy_score(real, decoded_predicted)))

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
#read in data using pandas
train_df = pd.read_csv("Web_Scraping/test.csv")

model = keras.models.load_model("model2.h5")
model.summary()
le = preprocessing.LabelEncoder()
Y = train_df['publication']
le.fit(Y)
string = train_df['content']
string = np.array(string)
predictions = model.predict(string, batch_size=32)
#predictions = decode(le, predictions)
HeatMap(Y, predictions, le)
print(decode(le, predictions))


#everything bellow doesnt matter
#Get X
X = list(train_df['content'])
#Get Y
Y = list(train_df['publication'])

#Preprocessing by converting string to int
le = preprocessing.LabelEncoder()
le.fit(Y)

x_enc = X
y_enc = encode(le, Y)

x_train, x_test, y_train, y_test = train_test_split(x_enc, y_enc, test_size=0.4, random_state=42)
#convert everything from list to array
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

predicts = model.predict(x_test, batch_size=32)

y_test = decode(le, y_test)
y_preds = decode(le, predicts)

#Creation of heatmap
y_unique = np.array(Y)
y_unique = np.unique(y_unique) #unique Y values
mat = confusion_matrix(y_true=y_test, y_pred=y_preds)
sns.heatmap(mat.T, square = True, annot=True, fmt = "d", xticklabels=y_unique,yticklabels=y_unique,normalize=normalize)
plt.xlabel("true labels")
plt.ylabel("predicted label")
plt.title("Predicted Values vs. Real Values")
plt.show()

#accuracy report based on given confusion matrix
print("The accuracy is {}".format(accuracy_score(y_test, y_preds)))