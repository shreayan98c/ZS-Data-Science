#!/usr/bin/env python
# coding: utf-8

# # Christiano Ronaldo Shot Predictor

# ### Import all the necessary header files as follows:

# **pandas** : An open source library used for data manipulation, cleaning, analysis and visualization. <br>
# **numpy** : A library used to manipulate multi-dimensional data in the form of numpy arrays with useful in-built functions. <br>
# **matplotlib** : A library used for plotting and visualization of data. <br>
# **seaborn** : A library based on matplotlib which is used for plotting of data. <br>
# **sklearn.metrics** : A library used to calculate the accuracy, precision and recall. <br>
# **sklearn.preprocessing** : A library used to encode and onehotencode categorical variables. 

# In[1]:


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import missingno as msno
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[2]:


# Importing the dataset
data = pd.read_csv("data.csv")


# ### Inspecting and cleaning the data

# In[3]:


# Printing the 1st 5 columns
data.drop(['Unnamed: 0'], axis=1, inplace=True)
data.head()


# In[4]:


# Printing the dimensions of data
data.shape


# In[5]:


# Viewing the column heading
data.columns


# In[6]:


# Inspecting the target variable
data['is_goal'].value_counts()


# ### Data Visualization

# In[7]:


msno.matrix(data)


# In[8]:


# Identifying the unique number of values in the dataset
data.nunique()


# In[9]:


# Imputing missing values


# In[10]:


data['match_event_id'] = data['match_event_id'].fillna(data['match_event_id'].mean())
data['location_x'] = data['location_x'].fillna(data['location_x'].mean())
data['location_y'] = data['location_y'].fillna(data['location_y'].mean())
data['remaining_min'] = data['remaining_min'].fillna(data['remaining_min'].mean())
data['power_of_shot'] = data['power_of_shot'].fillna(data['power_of_shot'].mean())
data['knockout_match'] = data['knockout_match'].fillna(-1)
data['game_season'] = data['game_season'].fillna('Unspecified')
data['remaining_sec'] = data['remaining_sec'].fillna(data['remaining_sec'].median())
data['distance_of_shot'] = data['distance_of_shot'].fillna(data['distance_of_shot'].mean())
data['area_of_shot'] = data['area_of_shot'].fillna('Unspecified')
data['shot_basics'] = data['shot_basics'].fillna('Unspecified')
data['range_of_shot'] = data['range_of_shot'].fillna('Unspecified')
data['home/away'] = data['home/away'].fillna('Unspecified')
data['type_of_shot'] = data['type_of_shot'].fillna('Unspecified')
data['remaining_min.1'] = data['remaining_min.1'].fillna(data['remaining_min.1'].median())
data['power_of_shot.1'] = data['power_of_shot.1'].fillna(data['power_of_shot.1'].median())
data['knockout_match.1'] = data['knockout_match.1'].fillna(-1)
data['remaining_sec.1'] = data['remaining_sec.1'].fillna(data['remaining_sec.1'].median())
data['distance_of_shot.1'] = data['distance_of_shot.1'].fillna(data['distance_of_shot.1'].mean())


# In[11]:


l = []
for item in data['home/away']:
    if '@' in item:
        l.append('Away')
    elif 'vs' in item:
        l.append('Home')
    else:
        l.append('Unspecified')
data['h/a'] = l


# In[12]:


def getlat(x):
    return float(str(x).split(',')[0])


# In[13]:


def getlong(x):
    try:
        r = str(x).split(',')[1]
        return float(r)
    except:
        return np.nan


# In[14]:


data['lat'] = data['lat/lng'].apply(getlat)


# In[15]:


data['long'] = data['lat/lng'].apply(getlong)


# In[16]:


data['lat'] = data['lat'].fillna(data['lat'].mean())
data['long'] = data['long'].fillna(data['long'].mean())


# In[17]:


data.head()


# In[18]:


# Dropping columns not needed for our model
data.drop(['match_event_id', 'lat/lng', 'team_name', 'home/away', 'team_id', 'date_of_game', 'match_id', 'type_of_combined_shot'], axis=1, inplace=True)


# In[19]:


data.dtypes


# In[20]:


data.isna().sum()


# In[21]:


train = data[data['is_goal'].notna()]


# In[22]:


train.drop(['shot_id_number'], axis=1, inplace=True)


# In[23]:


train.shape


# In[24]:


train.head()


# In[25]:


test = data[data['is_goal'].isna()]


# In[26]:


my_submission = pd.DataFrame({'shot_id_number': test['shot_id_number']})
my_submission.shape


# In[27]:


test.drop(['shot_id_number'], axis=1, inplace=True)


# In[28]:


test.shape


# In[29]:


test.head()


# In[30]:


corr = data.corr()


# In[31]:


plt.figure(figsize=(14,14))
sns.heatmap(corr, cbar=True, square= True, fmt='.2f',annot=True,annot_kws={'size':15}, cmap='Greens')


# In[32]:


final_data = pd.get_dummies(train)


# #### Once the data is cleaned, we split the data into training set and test set to prepare it for our machine learning model in a suitable proportion.

# In[33]:


# Spliting target variable and independent variables
X = final_data.drop(['is_goal'], axis = 1)
y = final_data['is_goal']


# In[34]:


# Splitting the data into training set and testset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0, stratify=y)


# ### Logistic Regression

# In[35]:


# Logistic Regression

# Import library for LogisticRegression
from sklearn.linear_model import LogisticRegression

# Create a Logistic regression classifier
logreg = LogisticRegression()

# Train the model using the training sets 
logreg.fit(X_train, y_train)


# In[36]:


# Prediction on test data
y_pred = logreg.predict(X_test)


# In[37]:


# Calculating the accuracy, precision and the recall
acc_logreg = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Total Accuracy : ', acc_logreg )
print( 'Precision : ', round( metrics.precision_score(y_test, y_pred) * 100, 2 ) )
print( 'Recall : ', round( metrics.recall_score(y_test, y_pred) * 100, 2 ) )


# In[38]:


# Create confusion matrix function to find out sensitivity and specificity
from sklearn.metrics import confusion_matrix
def draw_cm(actual, predicted):
    cm = confusion_matrix( actual, predicted, [1,0]).T
    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["Yes","No"] , yticklabels = ["Yes","No"] )
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.show()


# In[39]:


# Confusion matrix 
draw_cm(y_test, y_pred)


# ### Gaussian Naive Bayes

# In[40]:


# Gaussian Naive Bayes

# Import library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

# Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets 
model.fit(X_train,y_train)


# In[41]:


# Prediction on test set
y_pred = model.predict(X_test)


# In[42]:


# Calculating the accuracy, precision and the recall
acc_nb = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Total Accuracy : ', acc_nb )
print( 'Precision : ', round( metrics.precision_score(y_test, y_pred) * 100, 2 ) )
print( 'Recall : ', round( metrics.recall_score(y_test, y_pred) * 100, 2 ) )


# In[43]:


# Confusion matrix 
draw_cm(y_test, y_pred)


# ### Decision Tree Classifier

# In[44]:


# Decision Tree Classifier

# Import Decision tree classifier
from sklearn.tree import DecisionTreeClassifier

# Create a Decision tree classifier model
clf = DecisionTreeClassifier()

# Hyperparameter Optimization
parameters = {'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10, 50], 
              'min_samples_split': [2, 3, 50, 100],
              'min_samples_leaf': [1, 5, 8, 10]
             }

# Run the grid search
grid_obj = GridSearchCV(clf, parameters)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Train the model using the training sets 
clf.fit(X_train, y_train)


# In[45]:


# Prediction on training set
y_pred = clf.predict(X_train)


# In[46]:


# Finding the variable with more importance
feature_importance = pd.DataFrame([X_train.columns, clf.tree_.compute_feature_importances()])
feature_importance = feature_importance.T.sort_values(by = 1, ascending=False)[1:10]


# In[47]:


sns.barplot(x=feature_importance[1], y=feature_importance[0])
# Add labels to the graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# In[48]:


# Prediction on test set
y_pred = clf.predict(X_test)


# In[49]:


# Calculating the accuracy, precision and the recall
acc_dt = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Total Accuracy : ', acc_dt )
print( 'Precision : ', round( metrics.precision_score(y_test, y_pred) * 100, 2 ) )
print( 'Recall : ', round( metrics.recall_score(y_test, y_pred) * 100, 2 ) )


# In[50]:


# Confusion matrix 
draw_cm(y_test, y_pred)


# ### Random Forest Classifier

# In[51]:


# Random Forest Classifier

# Import library of RandomForestClassifier model
from sklearn.ensemble import RandomForestClassifier

# Create a Random Forest Classifier
rf = RandomForestClassifier()

# Hyperparameter Optimization
parameters = {'n_estimators': [4, 6, 9, 10, 15], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1, 5, 8]
             }

# Run the grid search
grid_obj = GridSearchCV(rf, parameters)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the rf to the best combination of parameters
rf = grid_obj.best_estimator_
# Train the model using the training sets 
rf.fit(X_train,y_train)


# In[52]:


# Finding the variable with more importance
feature_imp = pd.Series(rf.feature_importances_,index= X_train.columns).sort_values(ascending=False)
# Creating a bar plot
feature_imp=feature_imp[0:10,]
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to the graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# In[53]:


# Prediction on test data
y_pred = rf.predict(X_test)


# In[54]:


# Calculating the accuracy, precision and the recall
acc_rf = round( metrics.accuracy_score(y_test, y_pred) * 100 , 2 )
print( 'Total Accuracy : ', acc_rf )
print( 'Precision : ', round( metrics.precision_score(y_test, y_pred) * 100 , 2 ) )
print( 'Recall : ', round( metrics.recall_score(y_test, y_pred) * 100, 2 ) )


# In[55]:


# Confusion matrix 
draw_cm(y_test, y_pred)


# ### Support Vector Machine Classifier

# In[56]:


# SVM Classifier

# Creating scaled set to be used in model to improve the results
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[57]:


# Import Library of Support Vector Machine model
from sklearn import svm

# Create a Support Vector Classifier
svc = svm.SVC()

Hyperparameter Optimization
parameters = [
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['linear']}
]

# Run the grid search
grid_obj = GridSearchCV(svc, parameters)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the svc to the best combination of parameters
svc = grid_obj.best_estimator_

# Train the model using the training sets 
svc.fit(X_train,y_train)


# In[58]:


# Prediction on test data
y_pred = svc.predict(X_test)


# In[59]:


# Calculating the accuracy, precision and the recall
acc_svm = round( metrics.accuracy_score(y_test, y_pred) * 100, 2 )
print( 'Total Accuracy : ', acc_svm )
print( 'Precision : ', round( metrics.precision_score(y_test, y_pred) * 100, 2 ) )
print( 'Recall : ', round( metrics.recall_score(y_test, y_pred) * 100, 2 ) )


# In[60]:


# Confusion matrix 
draw_cm(y_test, y_pred)


# ### eXtreme Gradient Boosting

# In[61]:


# eXtreme Gradient Boosting

# Import library of xgboost model
from xgboost import XGBClassifier

# Create a XGB Classifier
gbm = XGBClassifier(learning_rate = 0.02, n_estimators= 2000, max_depth= 4, min_child_weight= 2, gamma=1, subsample=0.8, 
                    colsample_bytree=0.8, objective= 'binary:logistic', nthread= -1, scale_pos_weight=1)

# Train the model using the training sets 
gbm.fit(X_train, y_train)


# In[62]:


# Prediction on test data
y_pred = gbm.predict(X_test)


# In[63]:


# Calculating the accuracy, precision and the recall
acc_xgb = round( metrics.accuracy_score(y_test, y_pred) * 100 , 2 )
print( 'Total Accuracy : ', acc_xgb )
print( 'Precision : ', round( metrics.precision_score(y_test, y_pred) * 100 , 2 ) )
print( 'Recall : ', round( metrics.recall_score(y_test, y_pred) * 100, 2 ) )


# In[64]:


# Confusion matrix 
draw_cm(y_test, y_pred)


# ### Evaluation and comparision of all the models

# In[65]:


models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Naive Bayes', 'Decision Tree', 'Random Forest', 'Support Vector Machines', 'XGBoost'],
    'Score': [acc_logreg, acc_nb, acc_dt, acc_rf, acc_svm, acc_xgb]})
models.sort_values(by='Score', ascending=False)


# In[66]:


test_data = pd.get_dummies(test)


# In[67]:


test_data.drop(['is_goal'], axis=1, inplace=True)


# In[68]:


my_pred = logreg.predict(test_data)


# In[69]:


my_pred.shape


# In[70]:


my_submission['is_goal'] = my_pred


# In[71]:


my_submission.head()


# In[72]:


my_submission.shape


# In[74]:


# Saving predictions to the csv file
my_submission.to_csv('shreayan_chaudhary_060298_prediction_1.csv', index=False)
print ("File submitted successfully")

