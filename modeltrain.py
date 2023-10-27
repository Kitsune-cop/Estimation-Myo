import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns

attribute = pd.read_csv('./attribute_200_8.csv', header=None)
target = pd.read_csv('./target_200_8.csv', header=None)

attribute = np.array(attribute, dtype= np.float32)
attribute = np.reshape(attribute, (-1, 200, 8))
target = np.array(target, dtype= np.float32)

target.shape

# Splitting the data into 80% training set and 20% test set
X_train, X_test, y_train, y_test = train_test_split(attribute, target, test_size=0.2, random_state=1)
# Splitting the training set into 80% training subset and 20% validation set
X_train_subset, X_val, y_train_subset, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)


# import tensorflow as tf
# import tensorflow.keras as keras
# from  keras.layers import ConvLSTM1D, Dense
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
from keras.backend import dropout

model = tf.keras.Sequential([
    LSTM(50, return_sequences=True, dropout=0.2),
    LSTM(50, return_sequences=True, dropout=0.2),
    LSTM(50, return_sequences=True, dropout=0.2),
    LSTM(50),
    Dense(4, activation='softmax')
])


adam = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_subset,
        y_train_subset,
        validation_data=(X_val, y_val),
        batch_size=256,
        epochs=200)

y_pred = model.predict(X_test)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred.argmax(axis=1))

#creating data frame for a array-formatted confusion matrix, so it will be easy for plotting
cm_df = pd.DataFrame(cm,
                    index = ['Relax','Rock','Paper','Scissor'],
                    columns = ['Relax','Rock','Paper','Scissor'])

group_counts = ['{0:0.0f}'.format(value) for value in
                cm.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                    cm.flatten()/np.sum(cm)]
labels = [f'{v1}\n{v2}' for v1, v2 in
        zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(4,4)
# Print the confusion matrix
# print(cm)


import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

sns.heatmap(cm_df, annot=labels,fmt='', cmap='Blues')
plt.title('Confusion Matrix',fontsize=17)
plt.ylabel('Actual Values',fontsize=13)
plt.xlabel('Predicted Values',fontsize=13)
plt.show()

# sns.heatmap(cm_df,
#             annot=labels,
#             fmt='', cmap='Blues')
# plt.ylabel('Prediction',fontsize=13)
# plt.xlabel('Actual',fontsize=13)
# plt.title('Confusion Matrix',fontsize=17)
# plt.show()
# plt.savefig('./model_result/confuse_matrix_00_60.png')

acc_history = history.history['accuracy']
val_acc =  history.history['val_accuracy']

# Plot the acc history
plt.plot(acc_history)
plt.plot(val_acc)
plt.title('Accuracy History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train','test'], loc='lower right')
plt.show()
# plt.savefig('./model_result/acc_his_00_60.png')

# Access the loss history
loss_history = history.history['loss']
val_loss =  history.history['val_loss']

# Plot the loss history
plt.plot(loss_history)
plt.plot(val_loss)
plt.title('Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['train','test'], loc='upper right')
plt.show()
# plt.savefig('./model_result/loss_his_00_60.png')


# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)

print('Loss on test set: ', loss)
print('Accuracy on test set: ', accuracy)

classification_report = classification_report(y_test, y_pred.argmax(axis=1))
print(classification_report)

model.save('./model/model_60.keras')
