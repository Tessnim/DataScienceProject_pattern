import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import data
train_data = pd.read_csv('data/Train_v2.csv')
test_data = pd.read_csv('data/Test_v2.csv')

print('train data shape :', train_data.shape)
print('test data shape :', test_data.shape)

print(train_data.describe())
print(train_data.info())

# Check for missing values
print('missing values:', train_data.isnull().sum())

# Target distribution
train_data.bank_account.value_counts().plot(kind='bar')

from sklearn.preprocessing import LabelEncoder
# Convert target label to numerical Data
le = LabelEncoder()
train_data['bank_account'] = le.fit_transform(train_data['bank_account'])
train_data.head()

# Data visualisation
import seaborn as sns

f, axes = plt.subplots(7, 1, figsize=[25, 70])

sns.countplot('location_type', hue= 'bank_account', data=train_data, ax=axes[0])
sns.countplot('gender_of_respondent', hue= 'bank_account', data=train_data, ax=axes[1])
sns.countplot('cellphone_access', hue= 'bank_account', data=train_data, ax=axes[2])
sns.countplot('relationship_with_head', hue= 'bank_account', data=train_data, ax=axes[3])
sns.countplot('marital_status', hue= 'bank_account', data=train_data, ax=axes[4])
sns.countplot('education_level', hue= 'bank_account', data=train_data, ax=axes[5])
sns.countplot('job_type', hue= 'bank_account', data=train_data, ax=axes[6])

train_data['year_'] = train_data['year']
test_data['year_'] = test_data['year']
# Convert the following numerical labels from integer to float
float_array = train_data[['household_size', 'age_of_respondent', 'year_']].values.astype(float)
float_array = test_data[['household_size', 'age_of_respondent', 'year_']].values.astype(float)

# Data preprocessing
# convert categorical features to numerical features
# categorical features to be converted by One Hot Encoding
train_data['country_'] = train_data['country']
test_data['country_'] = test_data['country']

categ = ['relationship_with_head', 'marital_status', 'education_level', 'job_type', 'country_']
# One Hot Encoding conversion
train_data = pd.get_dummies(train_data, prefix_sep='_', columns = categ)

test_data = pd.get_dummies(test_data, prefix_sep='_', columns = categ)

# Labelncoder conversion
train_data['location_type'] = le.fit_transform(train_data['location_type'])
train_data['cellphone_access'] = le.fit_transform(train_data['cellphone_access'])
train_data['gender_of_respondent'] = le.fit_transform(train_data['gender_of_respondent'])


test_data['location_type'] = le.fit_transform(test_data['location_type'])
test_data['cellphone_access'] = le.fit_transform(test_data['cellphone_access'])
test_data['gender_of_respondent'] = le.fit_transform(test_data['gender_of_respondent'])


train_data.head()

test_data.head()

#Separate training features from target
X_train = train_data.drop(['year', 'uniqueid', 'bank_account', 'country'], axis=1)
y_train = train_data['bank_account']

X_test = test_data.drop(['year', 'uniqueid', 'country'], axis=1)

#rescale X_train and X_test
# import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_train_rescaled = scaler.fit_transform(X_train)
X_test_rescaled = scaler.fit_transform(X_test)

train_data.head()

X_train_rescaled.shape

# Split train_data
from sklearn.model_selection import train_test_split

X_Train, X_val, y_Train, y_val = train_test_split(X_train_rescaled, y_train, stratify = y_train, test_size = 0.2, random_state=42)

#import XGBClassifier
from xgboost import XGBClassifier

my_model = XGBClassifier()
# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Optimize model paramaters
# I run this code in google colab to make the execution much faster and use the best params in the next code
param_grid = {'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
my_model2 = GridSearchCV(my_model, param_grid)
my_model2.fit(X_Train, y_Train)
print(my_model2.best_params_)

from sklearn.metrics import confusion_matrix, accuracy_score

# fit and Evaluate model
my_model3 = XGBClassifier(min_child_weight = 1, gamma = 2, subsample = 0.6, colsample_bytree = 0.6, max_depth = 3)
my_model3.fit(X_Train, y_Train)
y_pred = my_model3.predict(X_val)

# Get error rate
print("Error rate of XGBoost: ", 1 - accuracy_score(y_val, y_pred))

# Get confusion matrix
confusion_matrix(y_pred, y_val)
