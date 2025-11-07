# Import necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
music_data = pd.read_csv('music.csv')

# Separate features (X) and target variable (y)
X = music_data.drop(columns=['genre'])  # Features: all columns except 'genre'
y = music_data['genre']                 # Target: 'genre' column

# Split the dataset into training and testing sets
# test_size=0.2 means 20% of data is used for testing, 80% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Decision Tree Classifier model
model = DecisionTreeClassifier()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict the genres for the test data
predictions = model.predict(X_test)

# Calculate the accuracy of the model
score = accuracy_score(y_test, predictions)

# Print the accuracy
print(f'Accuracy: {score}')
