from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Function to standardize data
def standardize_data(scaler, data, fit=False):
    if fit:
        return scaler.fit_transform(data)
    else:
        return scaler.transform(data)

# Load dataset
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize training and testing data
scaler = StandardScaler()
x_train_scaled = standardize_data(scaler, x_train, fit=True)  # Fit and transform training data
x_test_scaled = standardize_data(scaler, x_test)             # Transform test data

# Train and evaluate model on standardized data
model_scaled = SVC(kernel='rbf', gamma='scale')
model_scaled.fit(x_train_scaled, y_train)
y_pred_scaled = model_scaled.predict(x_test_scaled)
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)

# Train and evaluate model on raw data
model_raw = SVC(kernel='rbf', gamma='scale')
model_raw.fit(x_train, y_train)
y_pred_raw = model_raw.predict(x_test)
accuracy_raw = accuracy_score(y_test, y_pred_raw)

# Print results
print(f"Accuracy with Standardization: {accuracy_scaled:.4f}")
print(f"Accuracy without Standardization: {accuracy_raw:.4f}")
