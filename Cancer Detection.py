# Loading basic important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib auto

# Loading dataset from sklearn
from sklearn.datasets import load_breast_cancer

# Saving dataset into a variable
cancer = load_breast_cancer()

# Checking keys in dictionary
cancer.keys()

# Describing the dataest
print(cancer['DESCR'])
print(cancer['target'])
print(cancer['target_names'])
print(cancer['feature_names'])

# Checking the number of rows, columns of 'data' (Independent variables)
cancer['data'].shape

# Converting the data into pandas dataframe
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))

## Visualize the data
# Plotting a pairplot between variables for exploratory analysis
sns.pairplot (df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'])

# Plotting a plot to see the count
sns.countplot(df_cancer['target'])

#Plotting a scatterplot between area and smoothness
sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)

# Plotting a heatmap to see the correlation between variables
plt.figure(figsize = (20, 10))
sns.heatmap(df_cancer.corr(), annot = True)

# Dividing dataset into x and y (independent variables and dependent variables)
x = df_cancer.drop(['target'], axis = 1)
y = df_cancer['target']

# Loading train_test_split library from sklearn.model_selection
from sklearn.model_selection import train_test_split

# Dividing the dataset into X_train, X_test, y_train and y_test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 5)

# Importing SVC Classifier and Evaluation Metrics from sklearn
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Creating an object for the model
svc_model = SVC()

# Training the model
svc_model.fit(X_train, y_train)

# Testing model on Test dataset
y_predict = svc_model.predict(X_test)

# Comparing prediction values with actual values using confusion matrix
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot = True)

# Imporving the model

# Min Max Scaling on train and test dataset
min_train = X_train.min()
range_train = (X_train - min_train).max()
X_train_scaled = (X_train - min_train)/range_train

#Plotting a scatterplot between area and smoothness
sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = y_train)

min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test

# Training the model
svc_model.fit(X_train_scaled, y_train)

# Testing model on Test dataset
y_predict = svc_model.predict(X_test_scaled)

# Comparing prediction values with actual values using confusion matrix
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot = True)

# Printing classification report
print(classification_report(y_test, y_predict))


# Grid Search
param_grid = {'C' : [0.1, 1, 10, 100],
			  'gamma' : [1, 0.1, 0.01, 0.001],
			  'kernel' : ['rbf']}

# Loading GridSearchCV library from sklearn
from sklearn.model_selection import GridSearchCV

# Defining the GridSearch
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 4)

# Training our model with GridSearch
grid.fit(X_train_scaled, y_train)

# Finding the best tuning parameters
grid.best_params_

# Using the best tuning parameters to predict outcomes
grid_predictions = grid.predict(X_test_scaled)

# Calculating the Confusion Matrix
cm = confusion_matrix(y_test, grid_predictions)
cm

# Printing classification report
print(classification_report(y_test, grid_predictions))


## We were able to predict benign and malignant cancer with 96% accuracy and 0% Type II error.
## Lets make a Cancer Free World !