#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 10:42:26 2022

@author: sanjasrdanovic
"""
# Import packages and libraries

# pandas, matplotlib, seaborn, numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Scikit-Learn: 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
# Keras and Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam


# Check a version of tensorflow and keras
print(tf.__version__)
print(keras.__version__)

# Set tensorflow and numpy random seed for reproducible results
tf.random.set_seed(42)
np.random.seed(42)

#%%
# Loading Data:
data = pd.read_csv('diabetes.csv')

# seting printing values to see all data in the console
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', None)

# data info
print(data.head())
print(data.info())
# 768 entries, 9 columns
# 768 entries and 8 input features (variables) and 1 output feature
# print(data.dtypes)

#%%
"""First visualisations to get some ideas about the distribution of diabetes
   and correlation of variables
"""
# count plot for the Outcome - there is more women with no diabetes than with diabetes
sns.countplot(x='Outcome', data = data)
plt.show()
plt.savefig("Distribution_of_diabetes.png") # save figure   
# heatmap to check for the correlation
sns.heatmap(data.corr())
plt.show()
plt.savefig("Correlation_heatmap.png") # save figure   
# bar graph to check the correlation by Outcome and 
# get some ideas which variables have most influence
data.corr()['Outcome'][:-1].sort_values().plot(kind='bar')
plt.savefig("Correlation_barplot.png") # save figure   
# histogram for each variable
data.hist(figsize=(18,12))
plt.show()
plt.savefig("Histograms_variables.png") # save figure   

#%%

# Data Inspection

print(data.isnull().sum())
print(data.isna().sum())
print(data.describe())

# min values for Glucose, Blood Pressure, SkinThickness, Insulin, BMI cannot be 0
# => these are missing values
# label them as NaN values:

conditions_with_nan=['Glucose','BloodPressure','SkinThickness','Insulin','BMI']

for i in conditions_with_nan:
    print("There are " + str(len(data.loc[(data[i]==0),i])) + \
          " instances of value 0 in "+i+" variable.")

for i in conditions_with_nan:
    data.loc[(data[i]==0),i]=np.NaN

print('Missing Number of Observations for all Variables:' + \
      "\n" + str(data.isnull().sum()))
    
# to check that there are no more 0s as min values 
print(data[conditions_with_nan].min()) 

# data.dropna(inplace=True)
# print(data.info())
# only 392 entries left...
# in this article I read some of the ways to compensate for missing values in a dataset 
# https://towardsdatascience.com/6-different-ways-to-compensate-for-missing-values-data-imputation-with-examples-6022d9ca0779

# but I will first split data into training and test, and then impute mean values for training, 
# and for the test I will drop na-s and work with the real data that are left
#%%

# input values - 8 variables
# output data - outcome
X = data.iloc[:, 0:8].values  # Input values.
y = data.iloc[:, 8].values    # Output values (outcome 1-has diabetes, 0-no diabetes)

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=42)

print('X_train: ' + str(X_train.shape))
print('y_train: ' + str(y_train.shape))
print('X_test: ' + str(X_test.shape))
print('y_test: ' + str(y_test.shape))

print(type(X_test))
#%%
# Impute mean for the X_train
imp_mean = SimpleImputer(strategy='mean') #for median imputation replace 'mean' with 'median'
imp_mean.fit(X_train)
X_train = imp_mean.transform(X_train)

# drop the NaN values in X_test
i = pd.isnull(X_test).any(1).nonzero()[0]
y_test_fin=np.delete(y_test, i)

X_test = X_test[~np.isnan(X_test).any(axis=1)]
# print(X_test)

#%%
# scaling the data, i.e. normalize input features with MinMaxScaler
# This estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one.

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%%

# model 5
model = tf.keras.Sequential([
            tf.keras.layers.Dense(8, input_shape=(8,), activation='relu', kernel_initializer = 'he_normal'),
            tf.keras.layers.Dropout(0.25), # dropout for the input layer
            tf.keras.layers.Dense(32, activation='relu', kernel_initializer = 'he_normal'),
            tf.keras.layers.Dropout(0.5), # dropout for the hidden layer
            tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer = 'he_normal')
            ])

model.compile(optimizer='Adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])


# EarlyStopping to avoid overfitting
early_stopping = EarlyStopping(patience=30, monitor='val_loss', restore_best_weights=True)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test_fin), epochs=300, verbose = 1, callbacks=[early_stopping])

#%%

# Convert history of fitting to pandas Dataframe for plotting
history = pd.DataFrame(history.history)
# Plot losses and accuracy
history.plot(xlabel='epochs', ylabel='losses')
plt.show()


#%%
# Print out model architecture
print(model.summary())

#%%

# Final evaluation of generalisation error:
# Calculate accuracy on test set and print for final evaluation
acc_test = model.evaluate(X_test, y_test_fin)[1]
print(f'Accuracy Test Set      : {acc_test*1E2:.1f}%')

# print(X_test.shape)
# print(y_test.shape)
#%%

# compare y_test_fin and y_pred_cat
y_pred=model.predict(X_test)
y_pred_cat = np.around(y_pred)

#%%

# Confusion Matrix

cm = confusion_matrix(y_test_fin, y_pred_cat)
print(cm)

#%%
# Classification Report

modelrep = metrics.classification_report(y_test_fin,y_pred_cat)
print(modelrep)



