import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

train_file = os.path.join('data', 'train.tsv')
test_file = os.path.join('data', 'test.tsv')
results_file = os.path.join('data', 'results.tsv')
output_file = os.path.join('data', 'out.tsv')

df_test_names = ['Date',
                 'Temperature',
                 'Humidity',
                 'Light',
                 'CO2',
                 'HumidityRatio']

df_test = pd.read_csv(test_file, sep='\t', names=df_test_names)
df_test = df_test.dropna()

df_train_names = ['Occupancy',
                  'Date',
                  'Temperature',
                  'Humidity',
                  'Light',
                  'CO2',
                  'HumidityRatio']

df_train = pd.read_csv(train_file, sep='\t', names=df_train_names)
df_train = df_train.dropna()

# one variable
x_train = df_train[['Light']]
y_train = df_train.Occupancy

x_test = df_test[['Light']]
y_test = pd.read_csv(results_file, sep='\t', names=['Occupancy']).Occupancy


l_reg = LogisticRegression()
l_reg.fit(x_train, y_train)
y_train_pred = l_reg.predict(x_train)
y_test_pred = l_reg.predict(x_test)

train_conf_matrix = confusion_matrix(y_train, y_train_pred)
test_conf_matrix = confusion_matrix(y_test, y_test_pred)
tp_train, fp_train, fn_train, tn_train = train_conf_matrix.ravel()
tp_test, fp_test, fn_test, tn_test = test_conf_matrix.ravel()

# train_accuracy = accuracy_score(y_train, y_train_pred)
# train_accuracy = sum(y_train == y_train_pred) / len(y_train_pred)
train_accuracy = (tp_train+tn_train) / (sum(train_conf_matrix.ravel()))
train_sensitivity = tp_train / (tp_train + fn_train)
train_specificity = tn_train / (fp_train + tn_train)

# test_accuracy = accuracy_score(y_test, y_test_pred)
# test_accuracy = sum(y_test == y_test_pred) / len(y_test_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_sensitivity = tp_test / (tp_test + fn_test)
test_specificity = tn_test / (fp_test + tn_test)

# all variables
x_train_all = df_train[['Temperature',
                        'Humidity',
                        'Light',
                        'CO2',
                        'HumidityRatio']]
x_test_all = df_test[['Temperature',
                      'Humidity',
                      'Light',
                      'CO2',
                      'HumidityRatio']]

l_reg_all = LogisticRegression()
l_reg_all.fit(x_train_all, y_train)
y_train_pred_all = l_reg_all.predict(x_train_all)
y_test_pred_all = l_reg_all.predict(x_test_all)

train_conf_matrix_all = confusion_matrix(y_train, y_train_pred_all)
test_conf_matrix_all = confusion_matrix(y_test, y_test_pred_all)

tp_train_all, fp_train_all, fn_train_all, tn_train_all = \
    train_conf_matrix_all.ravel()

tp_test_all, fp_test_all, fn_test_all, tn_test_all = \
    test_conf_matrix_all.ravel()

train_accuracy_all = (tp_train_all+tn_train_all) \
                     / (sum(train_conf_matrix_all.ravel()))
train_sensitivity_all = tp_train_all / (tp_train_all + fn_train_all)
train_specificity_all = tn_train_all / (fp_train_all + tn_train_all)

test_accuracy_all = accuracy_score(y_test, y_test_pred_all)
test_sensitivity_all = tp_test_all / (tp_test_all + fn_test_all)
test_specificity_all = tn_test_all / (fp_test_all + tn_test_all)

pd.DataFrame(y_test_pred).to_csv(output_file, index=False, header=False)
