import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Load the dataset
file_path = 'path_to_your_file.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Step 2: Data Cleaning
# Drop rows with missing Severity (target variable)
df = df.dropna(subset=['Severity'])

# Fill missing values for numerical columns with median
df['Temperature(F)'].fillna(df['Temperature(F)'].median(), inplace=True)
df['Humidity(%)'].fillna(df['Humidity(%)'].median(), inplace=True)
df['Visibility(mi)'].fillna(df['Visibility(mi)'].median(), inplace=True)
df['Pressure(in)'].fillna(df['Pressure(in)'].median(), inplace=True)
df['Wind_Speed(mph)'].fillna(df['Wind_Speed(mph)'].median(), inplace=True)
df['Precipitation(in)'].fillna(0, inplace=True)  # Assume no precipitation if missing

# Drop columns with too many missing values or irrelevant information
df = df.drop(columns=['End_Lat', 'End_Lng', 'Number', 'Street', 'County', 'Zipcode',
                      'Airport_Code', 'Weather_Timestamp'])

# Step 3: Feature Engineering
# Extract time-based features from Start_Time
df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['Hour'] = df['Start_Time'].dt.hour
df['Day_of_Week'] = df['Start_Time'].dt.dayofweek
df['Month'] = df['Start_Time'].dt.month
df['Year'] = df['Start_Time'].dt.year

# Create a new feature 'Duration' as the difference between End_Time and Start_Time
df['End_Time'] = pd.to_datetime(df['End_Time'])
df['Duration'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60  # in minutes

# Fill missing categorical values with mode
df['Weather_Condition'].fillna(df['Weather_Condition'].mode()[0], inplace=True)
df['Sunrise_Sunset'].fillna(df['Sunrise_Sunset'].mode()[0], inplace=True)

# Convert categorical columns to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['Weather_Condition', 'Sunrise_Sunset', 'Side', 'State'], drop_first=True)

# Step 4: Prepare data for modeling
X = df.drop(columns=['ID', 'Severity', 'Start_Time', 'End_Time', 'Description', 'City', 'Country'])
y = df['Severity']

# Train-validation-test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)

# Step 5: Model Building
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Model Evaluation
# Validation set
y_pred_val = model.predict(X_val)
print("Validation Set Results:")
print(classification_report(y_val, y_pred_val))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_val))
print("Accuracy:", accuracy_score(y_val, y_pred_val))

# Test set
y_pred_test = model.predict(X_test)
print("\nTest Set Results:")
print(classification_report(y_test, y_pred_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
print("Accuracy:", accuracy_score(y_test, y_pred_test))

# Step 7: Recommendations
feature_importances = model.feature_importances_
features = X.columns
important_features = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
important_features = important_features.sort_values(by='Importance', ascending=False)

print("\nTop 10 Important Features:")
print(important_features.head(10))

# Plotting feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=important_features.head(10))
plt.title('Top 10 Important Features for Accident Severity Prediction')
plt.show()
