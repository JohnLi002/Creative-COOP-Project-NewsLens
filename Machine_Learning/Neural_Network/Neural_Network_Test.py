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


embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
#read in data using pandas
train_df = pd.read_csv("Datasets/test.csv")
#Check if csv was read correctly
print(train_df.head())

#Get X
X = list(train_df['content'])
#Get Y
Y = list(train_df['publication'])

#Preprocessing by converting string to int
le = preprocessing.LabelEncoder()
le.fit(Y)

x_enc = X
y_enc = encode(le, Y)

x_train, x_test, y_train, y_test = train_test_split(x_enc, y_enc, test_size=0.25, random_state=42)
#convert everything from list to array
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


def UniversalEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)))

#building model
input_text = Input(shape = (1,), dtype = tf.string)
embedding = Lambda(UniversalEmbedding, output_shape=(512,))(input_text)
dense = Dense(256, activation='relu') (embedding) #dense layer
dense2 = Dense(128, activation='relu') (dense) #extra test layer
pred = Dense(14, activation='softmax') (dense2) #prediction layer
model = Model(inputs=[input_text],outputs=pred)
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy']) #

model.summary()

#training model
history = model.fit(x_train, y_train, epochs=5, batch_size=128)
#saving model
model.save('./model.h5')

#model.load_model('./model.h5')
predicts = model.predict(x_test, batch_size=32)

y_test = decode(le, y_test)
y_preds = decode(le, predicts)

#Creation of heatmap
y_unique = np.array(Y)
y_unique = np.unique(y_unique) #unique Y values
mat = confusion_matrix(y_true=y_test, y_pred=y_preds,normalize='true')
sns.heatmap(mat.T, square = True, annot=True, fmt = ".4f", xticklabels=y_unique,yticklabels=y_unique)
plt.xlabel("true labels")
plt.ylabel("predicted label")
plt.title("Predicted Values vs. Real Values")
plt.show()

#accuracy report based on given confusion matrix
print("The accuracy is {}".format(accuracy_score(y_test, y_preds)))