import pandas as pd                         # For data manipulation
from sklearn.model_selection import train_test_split    # To split data into training and testing sets
from sklearn.linear_model import LogisticRegression     # For logistic regression modeling
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score      # For evaluating model performance
from sklearn.preprocessing import StandardScaler                         # For scaling and encoding
from sklearn.metrics import accuracy_score              #accuracy of model 
# Load the dataset from a CSV file
data = pd.read_csv("C:/Users/patel/Documents/gdsc/final_Model/dataset.csv")

# Dropping 'customerID' column as it's not useful for predictions
student = data.drop(['customerID'], axis=1)
# print(student.head())

# Convert 'TotalCharges' column to numeric type, replacing errors with NaN
student['TotalCharges'] = pd.to_numeric(student['TotalCharges'], errors='coerce')

# Converting  'Churn' to numeric 
X = student.drop('Churn', axis=1)               # Features (all columns except 'Churn')
y = student['Churn'].map({'Yes': 1, 'No': 0})   # Target variable (1 for 'Yes', 0 for 'No')

# One-Hot Encoding for categorical variables in features
X = pd.get_dummies(X)

# Standardize the features for Logistic Regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Fill any remaining missing values with the median
X.fillna(X.median(), inplace=True)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the Logistic Regression model
model = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)  # Fit the model on the training data

# Make predictions on the test set
y_predict = model.predict(X_test)

# Evaluate model performance
cm = confusion_matrix(y_test, y_predict)
print('\n--------------------Confusion Matrix:-------------------------\n', cm)

# Display classification report for detailed performance metrics
report = classification_report(y_test, y_predict)
print('----------------------------Classification Report:----------------------------------------\n', report)

# Calculate ROC AUC Score
prediction = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])  # Predict probabilities for ROC AUC
print('----------------------------Prediction Report:----------------------------------------\n')
print('ROC AUCCURACY Score:', prediction)
accuracy=accuracy_score(y_test,y_predict)
print("accuracy is :",accuracy)
#Accuracy -86%