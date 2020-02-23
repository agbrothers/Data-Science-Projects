from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam,SGD
from keras.utils import np_utils
from sklearn import metrics



""" DEFINE TRAIN/VALIDATION/TEST DATASETS """
data = pd.read_csv('preprocessed_audio.csv')
X = data.drop(columns=['Label'])
y = data['Label']

# 60 - 20 - 20 split of the data
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=1)



""" ENCODE LABELS """
encoder = LabelEncoder()
y_train = np_utils.to_categorical(encoder.fit_transform(Y_train))
y_val = np_utils.to_categorical(encoder.fit_transform(Y_val))
y_test = np_utils.to_categorical(encoder.fit_transform(Y_test))



""" CONTRUCT MODEL """
model = Sequential()
num_classes = y_train[0].shape[0]

model.add(Dense(256, input_shape=(257,)))
model.add(Activation('relu'))

model.add(Dense(256))
model.add(Activation('relu'))

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.fit(X_train, y_train, batch_size=10, epochs=60, validation_data=(X_val, y_val))

     

""" PRINT ACCURACY """          
result = model.predict(X_test)
correct = [(np.argmax(y_test[i])== np.argmax(result[i])) for i in range(len(y_test))]
acc = 100*sum(correct)/len(correct)
print("Accuracy: " + acc + "%")



""" PLOT CONFUSION MATRIX """
y_pred = [np.argmax(i) for i in result]
y_pred = pd.DataFrame(encoder.inverse_transform(y_pred))
conf_mat = confusion_matrix(Y_test, y_pred, labels=encoder.classes_)

fig, ax = plt.subplots(figsize=(8,8))
ax = sns.heatmap(conf_mat, annot=True, xticklabels=encoder.classes_, yticklabels=encoder.classes_)
ax.tick_params(labelsize=6)
plt.tight_layout()


































"""
ADDING TO NONE/UNKOWN CLASS
for i in range(len(y_test)):
    if(np.amax(result[i])<0.5):
      pred = 11
    else:
      pred = np.argmax(result[i])



# Visualize the distribution of samples by class
# Check that testing, training, and validation data follow the same distributions
rot = 50
fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(2,2,1)
ax.tick_params(labelsize=6)
plt.xticks(rotation=rot)
plt.title('data')
ax = plt.hist(sorted(list(data['Label'])), bins=9, align='left', rwidth=0.8, color='k')

ax1 = fig.add_subplot(2,2,2)
ax1.tick_params(labelsize=6)
plt.xticks(rotation=rot)
plt.title('y_train')
print('hi')
ax1 = plt.hist(sorted(list(y_train)), bins=9, align='left', rwidth=0.8, color='k')

ax2 = fig.add_subplot(2,2,3)
ax2.tick_params(labelsize=6)
plt.xticks(rotation=rot)
plt.title('y_test')
ax2 = plt.hist(sorted(list(y_test)), bins=9, align='left', rwidth=0.8, color='k')

ax3 = fig.add_subplot(2,2,4)
ax3.tick_params(labelsize=6)
plt.xticks(rotation=rot)
plt.title('y_val')
ax3 = plt.hist(sorted(list(y_val)), bins=9, align='left', rwidth=0.8, color='k')
"""





