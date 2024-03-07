import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow

# Load data
data = pd.read_csv('data.csv')

# Split data into features and target
X = data.drop('target', axis=1)
y = data['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Save model
joblib.dump(model, 'model.pkl')

# Log metrics and model to MLflow
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('my_experiment')

with mlflow.start_run():
    mlflow.log_metric('accuracy', accuracy)
    mlflow.sklearn.log_model(model, 'model')
