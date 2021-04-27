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
    mat = confusion_matrix(y_true=real, y_pred=decoded_predicted,normalize="true")
    sns.heatmap(mat.T, square = True, annot=True, fmt = ".4f", xticklabels=y_unique,yticklabels=y_unique)
    plt.xlabel("true labels")
    plt.ylabel("predicted label")
    plt.title("Predicted Values vs. Real Values")
    plt.show()
    #accuracy report based on given confusion matrix
    print("The accuracy is {}".format(accuracy_score(real, decoded_predicted))) #prints accuracy

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
#read in data using pandas
train_df = pd.read_csv("Datasets/test2.csv")

model = keras.models.load_model("model.h5") #loads previously trained model
model.summary()

#processes dataset
le = preprocessing.LabelEncoder()
Y = train_df['publication']
le.fit(Y)
content = train_df['content']
content = np.array(content)

#this is where the model begins to predict
print("Model Predicting")
predictions = model.predict(content, batch_size=32)
print("Predictions finished. Creating Heatmap")
HeatMap(Y, predictions, le)