
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
file_path = '/home/heemin/GFL/data_B_R0.csv'  # Update the file path as needed
data = pd.read_csv(file_path)

# Prepare the data for logistic regression
X = data.drop('malicious', axis=1)  # Features
y = data['malicious']  # Target variable

for seed in range(100):
    print("seed: ", seed)
    # Initialize StratifiedKFold for 5-fold split with a fixed random seed for reproducibility
    skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=seed)

    # Placeholder for accuracy scores for each fold
    accuracy_scores = []
    accuracy_dt = []

    # Iterate over each fold
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        # Split the data into training and test sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Fit the logistic regression model on the first fold only
        if i == 0:
            model = LogisticRegression(random_state=seed)
            dt = DecisionTreeClassifier(max_depth=1, random_state=seed)
            model.fit(X_test, y_test)
            dt.fit(X_test, y_test)

            # Display the coefficients and intercept
            coefficients = model.coef_[0]
            intercept = model.intercept_[0]
            coefficients_dict = {X.columns[i]: float(coefficients[i]) for i in range(len(coefficients))}
            regression_formula = f"malicious = {intercept:.2f} + " + " + ".join([f"({float(coeff):.2f} * {var})" for var, coeff in coefficients_dict.items()])
            print(regression_formula)
            # Display the decision tree
            print(dt.tree_.threshold)
            print(dt.tree_.value)
        
        else:
            # Make predictions on the test set and calculate accuracy
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)
            # Decision Tree
            y_pred_dt = dt.predict(X_test)
            accuracy_dt.append(accuracy_score(y_test, y_pred_dt))


    # Print the accuracy scores for each of the remaining folds
    print("Accuracy Scores:", accuracy_scores)
    print("Average Accuracy Score:", sum(accuracy_scores) / len(accuracy_scores))
    print("Accuracy Scores DT:", accuracy_dt)
    print("Average Accuracy Score DT:", sum(accuracy_dt) / len(accuracy_dt))
