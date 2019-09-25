import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from eda import X, Y
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X,Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
from sklearn.model_selection import train_test_split
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features

""" One hot encoding """
# One hot encode the variables
# X = pd.get_dummies(X)


""" split the data into train and test data """
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
X_train.shape[1]
X.info()

