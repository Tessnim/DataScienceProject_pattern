from eda import X, Y, numeric_dataframe
from sklearn.model_selection import train_test_split
import numpy as np

# Verify if X and Y are numeric (labelEncoding was applied earlier)
print(X.head())
print(Y.shape)
# split data into training and testing sets:
train_features, test_features, train_labels, test_labels = train_test_split(
    X,
    Y,
    test_size=0.32,
    random_state=41)

# get highly correlated features from correlation matrix:
correlated_features = set()
correlation_matrix = numeric_dataframe.corr()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)

# remove highly correlated features:
train_features.drop(labels=correlated_features, axis=1, inplace=True)
test_features.drop(labels=correlated_features, axis=1, inplace=True)

print(train_features.shape, test_features.shape)

###################################### RandomForestClassifier ##########################################

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from mlxtend.feature_selection import SequentialFeatureSelector

feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1),
                                             k_features=7,
                                             forward=True, verbose=2, scoring='roc_auc', cv=4)

features = feature_selector.fit(np.array(train_features.fillna(0)), train_labels)

filtered_features= train_features.columns[list(features.k_feature_idx_)]
print(filtered_features)

# see result of RandomForestClassifier:

clf = RandomForestClassifier(n_estimators=100, random_state=41, max_depth=3)
clf.fit(train_features[filtered_features].fillna(0), train_labels)
print("RandomForestClassifier")
train_pred = clf.predict_proba(train_features[filtered_features].fillna(0))
print('Accuracy on training set: {}'.format(roc_auc_score(train_labels, train_pred[:,1])))

test_pred = clf.predict_proba(test_features[filtered_features].fillna(0))
print('Accuracy on test set: {}'.format(roc_auc_score(test_labels, test_pred [:,1])))

############################ XGBoost-SequentialFeatureSelector-forward ####################################

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

from mlxtend.feature_selection import SequentialFeatureSelector

feature_selector = SequentialFeatureSelector(XGBClassifier(),
                                             k_features=7,
                                             forward=False, verbose=2, scoring='roc_auc', cv=4)

features = feature_selector.fit(np.array(train_features.fillna(0)), train_labels)

filtered_features = train_features.columns[list(features.k_feature_idx_)]
print(filtered_features)

# see result of XGBoost:

clf = XGBClassifier()
clf.fit(train_features[filtered_features].fillna(0), train_labels)
print("XGBoost")
train_pred = clf.predict_proba(train_features[filtered_features].fillna(0))
print('Accuracy on training set: {}'.format(roc_auc_score(train_labels, train_pred[:, 1])))

test_pred = clf.predict_proba(test_features[filtered_features].fillna(0))
print('Accuracy on test set: {}'.format(roc_auc_score(test_labels, test_pred[:, 1])))

##################################### XGBoost - Exhaustive feature Selector ############################################

from mlxtend.feature_selection import ExhaustiveFeatureSelector
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

feature_selector = ExhaustiveFeatureSelector(XGBClassifier(), min_features=2, max_features=10, scoring='roc_auc',
                                             print_progress=True,
                                             cv=2)

features = feature_selector.fit(np.array(train_features.fillna(0)), train_labels)

print(type(features))
filtered_features = train_features.columns[list(features.best_idx_)]
print(filtered_features)

# see result of XGBoost:

clf = XGBClassifier(min_child_weight = 5, gamma = 0.5, subsample = 0.6, colsample_bytree = 0.6, max_depth = 5)
clf.fit(train_features[filtered_features].fillna(0), train_labels)
print("XGBoost")
train_pred = clf.predict_proba(train_features[filtered_features].fillna(0))
print('Accuracy on training set: {}'.format(1- roc_auc_score(train_labels, train_pred[:, 1])))

test_pred = clf.predict_proba(test_features[filtered_features].fillna(0))
print('Accuracy on test set: {}'.format(1- roc_auc_score(test_labels, test_pred[:, 1])))