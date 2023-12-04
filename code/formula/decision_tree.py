from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.tree import export_graphviz
import graphviz

import pandas as pd

# Load the provided data
file_path = '/home/heemin/GFL/data_p060.csv'
data = pd.read_csv(file_path)

# Separating features and target variable
X = data.drop('malicious', axis=1)
y = data['malicious']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating a Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()

# Training the model
dt_classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = dt_classifier.predict(X_test)

# Evaluating the model
classification_report_output = classification_report(y_test, y_pred)

classification_report_output

# Exporting the decision tree as a dot file
dot_data = export_graphviz(dt_classifier, out_file=None, 
                           feature_names=X.columns,  
                           class_names=['Non-Malicious', 'Malicious'],
                           filled=True, rounded=True, 
                           special_characters=True)

# Generating the graph from dot data
graph = graphviz.Source(dot_data)

# Saving the graph to a PNG file
png_graph_file_path = '/home/heemin/GFL/data_p060'
graph.format = 'png'
graph.render(png_graph_file_path)