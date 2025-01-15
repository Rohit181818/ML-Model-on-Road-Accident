# ML-Model-on-Road-Accident
Documentation for Junior Machine Learning Engineer Assignment

Assignment Overview

The goal of this assignment is to analyze a road accidents dataset from the United States, develop a machine learning model to predict accident severity, and provide recommendations based on insights obtained from the data. The task involves data cleaning, feature engineering, exploratory data analysis, model building, and evaluation.

Steps Followed

1. Data Loading

The dataset was provided as a CSV file containing various features related to traffic accidents.
It was loaded using pandas for further analysis.

2. Data Cleaning

Handling Missing Values:
Dropped rows with missing values in critical columns like Severity.
For numerical columns such as Temperature(F), Humidity(%), and Visibility(mi), missing values were filled using the median.
Categorical columns with missing values were filled using the mode.
Dropped columns with excessive missing data such as End_Lat, End_Lng, Airport_Code, and Weather_Timestamp.

3. Feature Engineering

Time-based Features:
Extracted Hour, Day_of_Week, Month, and Year from the Start_Time column.
Calculated Duration as the time difference between Start_Time and End_Time in minutes.
One-Hot Encoding:
Applied one-hot encoding to categorical features like Weather_Condition, Sunrise_Sunset, and State.

4. Data Splitting

Split the dataset into:
Training Set: 75%
Validation Set: 15%
Test Set: 10%

5. Model Building

Model Used:
A RandomForestClassifier was chosen for its robustness and ability to handle both numerical and categorical features.
The model was trained on the training set.

6. Model Evaluation

Evaluated the model on both validation and test sets using:
Accuracy Score
Confusion Matrix
Classification Report
The model achieved a good accuracy score, and the classification report provided detailed precision, recall, and F1-score for each class.

7. Feature Importance Analysis

Extracted feature importance from the Random Forest model.
Identified the top 10 most important features contributing to accident severity prediction.
Visualized the feature importance using a bar plot.

8. Recommendations
Based on the analysis and feature importance, the following recommendations were made to improve road safety:
Monitoring Weather Conditions: Since weather plays a significant role in accident severity, deploying real-time weather monitoring systems could help in issuing timely alerts.
Enhancing Road Infrastructure: Features like Junction, Traffic_Signal, and Crossing significantly impact accident severity. Improving road infrastructure at critical points could mitigate severe accidents.
Time-based Strategies: Implementing time-based strategies such as increased patrolling during high-risk hours could help reduce accidents.