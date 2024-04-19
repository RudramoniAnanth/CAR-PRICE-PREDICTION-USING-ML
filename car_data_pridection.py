import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression

# Load the data
data = pd.read_csv("car data.csv")

# Create the feature matrix (X) and target variable (y)
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

# Encode categorical variables using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = LinearRegression()

# Calculate metrics without feature selection
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)
accuracy_train = model.score(X_train, y_train)
print("~~~~~~~~~~: Fold wise evaluation Metrics :~~~~~~~~~~")
print()
print("Foldwise MSE without feature selection on training validation:", mse_train)
print("Foldwise R2 without feature selection on training validation:", r2_train)
print("Foldwise Accuracy without feature selection on training validation:", accuracy_train)
print("-------------------------------------------------------------")

coef = model.coef_
intercept = model.intercept_
print("Linear Regression Equation:")
print()

print("y = {:.2f}".format(intercept), end=" ")

for i in range(len(coef)):
    print("+ {:.2f} * X{}".format(coef[i], X.columns[i]), end=" ")

print()

# Feature selection using ANOVA
selector = SelectKBest(score_func=f_regression, k=min(X.shape[1], 5))
selector.fit(X, y)
selected_feature_indices = selector.get_support(indices=True)
selected_feature_names = X.columns[selected_feature_indices]
X_train_selected = X_train[selected_feature_names]
X_test_selected = X_test[selected_feature_names]

# Calculate metrics with feature selection
model.fit(X_train_selected, y_train)
y_pred_train_selected = model.predict(X_train_selected)
mse_train_selected = mean_squared_error(y_train, y_pred_train_selected)
r2_train_selected = r2_score(y_train, y_pred_train_selected)
accuracy_train_selected = model.score(X_train_selected, y_train)
print()
print("~~~~~~~~~~: Evaluation Metrics  on training validation:~~~~~~~~~~")
print()
print("Foldwise MSE with feature selection on training validation:", mse_train_selected)
print("Foldwise R2 with feature selection on training validation:", r2_train_selected)
print("Foldwise Accuracy with feature selection on training validation:", accuracy_train_selected)
print("-------------------------------------------------------------")

coef = model.coef_
intercept = model.intercept_
print("Linear Regression Equation:")
print()
print("y = {:.2f}".format(intercept), end=" ")
for i in range(len(coef)):
    print("+ {:.2f} * X{}".format(coef[i], selected_feature_names[i]), end=" ")


# Evaluate model on the test set
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
accuracy_test = model.score(X_test, y_test)
print()
print()
print("~~~~~~~~~~: Evaluation Metrics without feature selection:~~~~~~~~~~")
print()
print("Testing MSE without feature selection:", mse_test)
print("Testing R2 without feature selection:", r2_test)
print("Testing Accuracy without feature selection:", accuracy_test)
print("-------------------------------------------------------------")

model.fit(X_train_selected, y_train)
y_pred_test_selected = model.predict(X_test_selected)
mse_test_selected = mean_squared_error(y_test, y_pred_test_selected)
r2_test_selected = r2_score(y_test, y_pred_test_selected)
accuracy_test_selected = model.score(X_test_selected, y_test)
print("~~~~~~~~~~: Evaluation Metrics with feature selection:~~~~~~~~~~")
print()
print("Testing MSE with feature selection:", mse_test_selected)
print("Testing R2 with feature selection:", r2_test_selected)
print("Testing Accuracy with feature selection:", accuracy_test_selected)
print("-------------------------------------------------------------")