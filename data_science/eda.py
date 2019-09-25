# imports:
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

######################   Get the Data  ########################################
dataframe = pd.read_csv("data/Train_v2.csv")
dataframe = dataframe.drop(["uniqueid"], axis=1) # drop unique_id (won't help analyzing data)
test_data = pd.read_csv("data/Test_v2.csv")

###################### Exploratory Data Analysis ##############################
dataframe.info()
dataframe.head()
dataframe.shape
dataframe.columns
dataframe.index
dataframe.describe()
dataframe["marital_status"].value_counts() # bank_account = target

# plotting target with countplot:
np.unique(dataframe["bank_account"], return_index=False, return_inverse=False, return_counts=True, axis=None)
sns.countplot(dataframe["bank_account"],label="Count")

# Boxplots:
# Boxplot of bank_account by location_type
# Doesn't help for binary target
trace0 = go.Box(
    y=dataframe.loc[dataframe['location_type'] == 'Rural']['bank_account'],
    name = 'Rural location',
    marker = dict(
        color = 'rgb(214, 12, 140)')
)
trace1 = go.Box(
    y=dataframe.loc[dataframe['location_type'] == 'Urban']['bank_account'],
    name = 'Urban location',
    marker = dict(
        color = 'rgb(0, 128, 128)',
    )
)
data = [trace0, trace1]
layout = go.Layout(
    title = "Boxplot of having a bank_account by location type"
)

fig = go.Figure(data=data,layout=layout)
iplot(fig)


# Boxplot of age by location_type
# Doesn't help for binary target
trace0 = go.Box(
    y=dataframe.loc[dataframe['location_type'] == 'Rural']['age_of_respondent'],
    name = 'Rural location',
    marker = dict(
        color = 'rgb(214, 12, 140)')
)
trace1 = go.Box(
    y=dataframe.loc[dataframe['location_type'] == 'Urban']['age_of_respondent'],
    name = 'Urban location',
    marker = dict(
        color = 'rgb(0, 128, 128)',
    )
)
data = [trace0, trace1]
layout = go.Layout(
    title = "Boxplot of age_of_respondent by location type"
)

fig = go.Figure(data=data,layout=layout)
iplot(fig)



# Boxplots:
# Boxplot of age by location_type
# Doesn't help for binary target
trace0 = go.Box(
    y=dataframe.loc[dataframe['marital_status'] == 'Married/Living together']['age_of_respondent'],
    name = 'Married/Living together',
    marker = dict(
        color = 'rgb(214, 12, 140)')
)
trace1 = go.Box(
    y=dataframe.loc[dataframe['marital_status'] == 'Single/Never Married']['age_of_respondent'],
    name = 'Single/Never Married',
    marker = dict(
        color = 'rgb(0, 128, 128)',
    )
)
trace2 = go.Box(
    y=dataframe.loc[dataframe['marital_status'] == 'Widowed']['age_of_respondent'],
    name = 'Widowed',
    marker = dict(
        color = 'rgb(173, 184, 75)')
)
trace3 = go.Box(
    y=dataframe.loc[dataframe['marital_status'] == 'Divorced/Seperated']['age_of_respondent'],
    name = 'Divorced/Seperated',
    marker = dict(
        color = 'rgb(21, 0, 252)')
)
trace4 = go.Box(
    y=dataframe.loc[dataframe['marital_status'] == 'Dont know']['age_of_respondent'],
    name = 'Dont know',
    marker = dict(
        color = 'rgb(237, 9, 9)')
)
data = [trace0, trace1, trace2, trace3, trace4]
layout = go.Layout(
    title = "Boxplot of age_of_respondent by marital status"
)

fig = go.Figure(data=data,layout=layout)
iplot(fig)



dataframe.groupby('marital_status')['bank_account'].describe()




# histogram of all features:
dataframe.hist(bins=50, figsize=(20, 20))
plt.show()

# label encoding for the dataframe
from sklearn.preprocessing import LabelEncoder


class MultiColumnLabelEncoder:

    def __init__(self, columns=None):
        self.columns = columns  # list of column to encode

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

# Separating target from data
X = dataframe.drop(["bank_account"], axis=1)
Y = dataframe.bank_account


labelencoder_X = MultiColumnLabelEncoder()
X = labelencoder_X.fit_transform(X)
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


f = plt.figure(figsize=(19, 15))
plt.matshow(X.corr(), fignum=f.number)
plt.xticks(range(X.shape[1]), X.columns, fontsize=14, rotation=45)
plt.yticks(range(X.shape[1]), X.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)



# plt.figure(figsize=(19,15))
# matrix = sns.heatmap(X.corr())
# cmap = sns.diverging_palette(5, 250, as_cmap=True)

# def magnify():
#     return [dict(selector="th",
#                  props=[("font-size", "7pt")]),
#             dict(selector="td",
#                  props=[('padding', "0em 0em")]),
#             dict(selector="th:hover",
#                  props=[("font-size", "12pt")]),
#             dict(selector="tr:hover td:hover",
#                  props=[('max-width', '200px'),
#                         ('font-size', '12pt')])
# ]

# matrix.style.background_gradient(cmap, axis=1)\
#     .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
#     .set_caption("Hover to magify")\
#     .set_precision(2)\
#     .set_table_styles(magnify())
# didn't work


labelencoder_X = MultiColumnLabelEncoder()
numeric_dataframe = labelencoder_X.fit_transform(dataframe)

