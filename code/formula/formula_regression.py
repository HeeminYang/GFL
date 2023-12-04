import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression

# Load your data
data = pd.read_csv('/home/heemin/GFL/data_B_R.csv')

# Define your independent variables (X) and dependent variable (y)
X = data.drop('malicious', axis=1)
y = data['malicious']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create a logistic regression model
model = LogisticRegression(max_iter=1000)

# Fit the model with the new dataset
model.fit(X, y)
# Fit the model with the training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# # Create a linear regression model
# model = LogisticRegression(max_iter=1000)

# # Fit the model with the entire dataset
# model.fit(X, y)

# Get the coefficients and the intercept of the model
coefficients = model.coef_
intercept = model.intercept_

# 독립 변수의 이름과 계수를 연결 (이번에는 각 계수를 float으로 변환)
coefficients_dict = {X.columns[i]: float(coefficients[0][i]) for i in range(len(coefficients[0]))}

# 회귀식 생성 (계수를 float으로 형식화)
regression_formula = f"malicious = {intercept[0]:.2f} + " + " + ".join([f"({coeff:.2f} * {var})" for var, coeff in coefficients_dict.items()])

print("회귀식:")
print(regression_formula)