import pandas as pd                                         # For data manipulation
import matplotlib.pyplot as plt                             # For plotting graphs
from sklearn.model_selection import train_test_split        # To split data into training and testing sets
from sklearn.ensemble import RandomForestClassifier         # For Random Forest classification
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score  # For evaluating model performance          
from sklearn.metrics import accuracy_score                  #accuracy of model 

# Load the dataset from a CSV file
data = pd.read_csv("C:/Users/patel/Documents/gdsc/final_Model/dataset.csv")
data.isnull().sum()                 #check for null values 

# Preprocessing: Preparing features (X) and target variable (y) for modeling
X = data.drop(['customerID','Churn'], axis=1)  # Features (all columns except 'Churn')
y = data['Churn'].map({'Yes': 1, 'No': 0})  # Target variable (1 for 'Yes', 0 for 'No')

# One-Hot Encoding for categorical variables in features
X = pd.get_dummies(X)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')  # Create the model with 100 trees
model.fit(X_train, y_train)  # Fit the model on the training data

# Make predictions on the test set
y_predict = model.predict(X_test)

# Evaluate model performance using a confusion matrix
cm = confusion_matrix(y_test, y_predict)
print('\n--------------------------Confusion Matrix:---------------------------------------------\n', cm)

# Display classification report for detailed performance metrics
report = classification_report(y_test, y_predict)
print('----------------------------Classification Report:----------------------------------------\n', report)

accuracy = accuracy_score(y_test, y_predict)
print('----------------------------Prediction  Report:-------------------------------------------\n')
print('Accuracy:           ', accuracy)        #while ROC AUC shows how well the model distinguishes between the different classes."

# Calculate ROC AUC Score
prediction = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])  # Predict probabilities for ROC AUC
print('ROC AUC Score:      ', prediction)

#79%