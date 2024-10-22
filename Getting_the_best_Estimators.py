
# coding: utf-8

# In[178]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#get_ipython().magic(u'matplotlib inline')
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier


# # Reading the Dataset
# 
# Reading the transformed dataset and separating the target variable and predictor variables

# In[366]:

data=pd.read_csv('data1.csv')


# In[367]:

data=data.set_index(['Date'])


# In[368]:

X=data.drop('Failure',axis=1)
X=X.reset_index().drop('Date',axis=1)
Y=data.Failure


# In[369]:

Y=Y.reset_index().drop('Date',axis=1)


# # Getting the best predictors using RandomForest Classifier
# I am running RandomForest Classification algorithm on whole dataset to fetch the best predicting variable.I am running the classification using grid search to get the best parameters for this model

# In[370]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score,recall_score,roc_auc_score
from sklearn.grid_search import GridSearchCV


# In[ ]:




# In[371]:

clf = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 6, 9,10,15,25], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(recall_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X,data.Failure)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
clf.fit(X,data.Failure)


# Getting the recall score for this Model.I am just running this model to get best estimators.So am running and validating on whole data instead train and test data.Hence this score does not have any significance

# In[372]:

predictions = clf.predict(X)
#print(recall_score(Y.Failure, predictions))


# # Listing the feature importance

# In[373]:

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(10):
    print("%d. fea1ure %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))


# # Visualization of best Estimators based on Random Forest

# In[374]:

top=[]
topcols=[]
for f in range(len(importances)):
    top.append(importances[indices[f]])
    topcols.append(X.columns[indices[f]])



plt.figure( figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.title("Feature importances")
plt.barh(range(10), top[0:10], yerr=std[top[0:10]],color='r', align="center")
plt.yticks(range(10), X.columns[indices[:10]])
plt.ylim([-1, 10])
#f1.show()


# # Getting the best estimators using the correlation functions
# For feature selection we look at correlation of each feature with the target variable.I used the below algorithms for getting correlations
# 
# Spearman-Rank correlation for nominal vs nominal data
# 
# Point-Biserial correlation for nominal vs continuous data

# In[358]:

from scipy.stats import pointbiserialr, spearmanr
columns = data.columns.values

param=[]
correlation=[]
abs_corr=[]

for c in columns:
    #Check if binary or continuous
    if len(data[c].unique())<=2:
        corr = spearmanr(data['Failure'],data[c])[0]
    else:
        corr = pointbiserialr(data['Failure'],data[c])[0]
    param.append(c)
    correlation.append(corr)
    abs_corr.append(abs(corr))

#Create dataframe for visualization
param_df=pd.DataFrame({'correlation':correlation,'parameter':param, 'abs_corr':abs_corr})

#Sort by absolute correlation
param_df=param_df.sort('abs_corr',ascending=False)

#Set parameter name as index
param_df=param_df.set_index('parameter')
print "Best Features using correlation function"
print param_df.head(10)


# Plotting the accuracy score with number of features selected in the model

import sklearn
scoresCV = []
scores = []
topcols.insert(0,'Failure')
for i in range(1,len(topcols)):
    new_df=data[topcols[0:i+1]]
    X = new_df.ix[:,1::]
    y = new_df.ix[:,0]
    scoreCV = sklearn.cross_validation.cross_val_score(clf, X, y, cv= 10)
    scores.append(np.mean(scoreCV))
    
plt.figure(figsize=(15,5))
plt.plot(range(1,len(scores)+1),scores, '.-')
plt.axis("tight")
plt.title('Feature Selection', fontsize=14)
plt.xlabel('# Features', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.grid();
plt.show();






