#
# import time
# from datetime import datetime
# from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score,precision_score,recall_score
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.datasets import load_iris
# import seaborn as sns
#
# def main():
#     X,y = load_iris(return_X_y=True)
#     print(X.shape)
#     print(y.shape)
#     X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
#     print(X_train.shape,X_test.shape)
#     print(y_train.shape,y_test.shape)
#     print(X)
#     logit = LogisticRegression(max_iter=1000,solver='liblinear')
#     logit.fit(X_train,y_train)
#
#     print(f'Score = {logit.score(X_test,y_test)}')
#     y_pred = logit.predict(X_test)
#     print(classification_report(y_test,y_pred))
#
# if __name__ == '__main__':
#     main()

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

# Load the iris dataset
iris = load_iris()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Create a logistic regression model and fit it to the training data
logreg = LogisticRegression(max_iter=5000,solver='saga',penalty='l2',multi_class='ovr')
logreg.fit(X_train, y_train)

# Use the model to make predictions on the testing data
y_pred = logreg.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")

# Use cross-validation to evaluate the model's performance
scores = cross_val_score(logreg, iris.data, iris.target, cv=10)

print(f"Cross-validation scores: {scores}")
print(f"Mean cross-validation score: {scores.mean()}")