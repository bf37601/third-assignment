import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

df_public = pd.read_csv('survey_results_public.csv',
                        usecols=['Respondent',
                                 'Hobbyist',
                                 'YearsCode',
                                 'YearsCodePro',
                                 'BetterLife'],
                        index_col='Respondent')

df_public.dropna(inplace=True)

df_public.replace(to_replace='Less than 1 year', value='0', inplace=True)
df_public.replace(to_replace='More than 50 years', value='50', inplace=True)
df_public.replace(to_replace='Yes', value='1', inplace=True)
df_public.replace(to_replace='No', value='0', inplace=True)

df_public = df_public.astype(int)

x_train, x_test, y_train, y_test = train_test_split(df_public[['Hobbyist', 'YearsCode', 'YearsCodePro']], df_public.BetterLife, random_state=20)

l_reg = LogisticRegression()
l_reg.fit(x_train, y_train)
l_reg.fit(x_test, y_test)
y_train_pred = l_reg.predict(x_train)
y_test_pred = l_reg.predict(x_test)

# train
train_conf_matrix = confusion_matrix(y_train, y_train_pred)
tp_train, fp_train, fn_train, tn_train = train_conf_matrix.ravel()

train_accuracy = (tp_train+tn_train) / (sum(train_conf_matrix.ravel()))
train_sensitivity = tp_train / (tp_train + fn_train)
train_specificity = tn_train / (fp_train + tn_train)

f1_train = f1_score(y_train, y_train_pred)

# test
test_conf_matrix = confusion_matrix(y_test, y_test_pred)
tp_test, fp_test, fn_test, tn_test = test_conf_matrix.ravel()

test_accuracy = (tp_test+tn_test) / (sum(test_conf_matrix.ravel()))
test_sensitivity = tp_test / (tp_test + fn_test)
test_specificity = tn_test / (fp_test + tn_test)

f1_test = f1_score(y_test, y_test_pred)
