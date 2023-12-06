### IMPORTS ###
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

### CONSTANTS ###
CV = 10 # Cross Validation folds
TS = 0.2 # Test Size
RS = 18 # Random State for reproducibility

### DATA SETUP ###

# Read in dataset
data = pd.read_csv("heart-anomalies.csv")

# Select feature columns
X = data.iloc[:, 1:]  
# Select target column
y = data.iloc[:, 0]   

# Separate the training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TS, random_state=RS)

# Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=RS)

### CROSS VALIDATION ###

# Cross Validation
cv_scores = cross_val_score(clf, X_train, y_train, cv=CV) 
print("CV Scores: ", cv_scores)
print(str(CV) + " fold average CV Score: ", np.mean(cv_scores))

### TRAIN AND TEST ###

# Train the classifier and evaluate it on the test set
clf.fit(X_train, y_train)
test_score = clf.score(X_test, y_test)
print("Test Set Score: ", test_score)