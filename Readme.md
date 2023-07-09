Homesite Quote Conversion Prediction

Homesite Quote Conversion Prediction is a Kaggle competition posten on 09-02-2016. Homesite is a leading provider of homeowners insurance. The task is to predict which customer will purchase a given quote using an anonymized database of information on customer and sales activity, including property and coverage information.

Importing Libraries:
The code begins by importing the required libraries for data manipulation, analysis, and modeling.

Mounting Google Drive:
The code mounts the Google Drive to access the necessary data files.

Loading Data:
The code loads the training, test, and sample submission data from the specified file paths.
It also extracts the target variable (y_train) from the training data and prepares the modified version of the test target variable (y_test_modified).

Data Visualization:
The code includes data visualization using the matplotlib and seaborn libraries.
It generates a bar plot to display the count of each class in the target variable.
It creates a bar plot to visualize the correlation between the top 50 features and the target variable.
It generates a heatmap to visualize the correlation between the top important features.
It displays the correlation values between the top 25 correlated features.
It generates various plots and prints information about categorical features.

Data Analysis:
The code performs analysis on specific features and their relationships with the target variable.
It calculates and displays information about the feature 'PropertyField29'.
It displays the top features with more than 7 unique values.
It analyzes the feature 'Field6' and displays the count of ones and zeros based on its values.
It visualizes the distribution of ones count and zeros count by Field6 using a bar plot.
It calculates and displays the variances of features and identifies features with low variance.
It calculates and displays the count of unique values in each feature.
The code performs additional tasks like extracting field categories from column names and saving mean and standard deviation values to pickle files.

Preprocessing:
The code defines a preprocess() function that applies specific transformations to the data.
It replaces commas in the 'Field10' column and strips and replaces empty values in certain columns with the most frequent value.
It fills missing values in selected columns with their respective most frequent values.
It handles missing values in the 'PersonalField84' column differently based on the presence of the 'QuoteConversion_Flag' column.
The function returns the preprocessed data.

Featurization:
The code defines a featurize() function that performs feature engineering on the data.
It creates new features based on the date ('Original_Quote_Date') column, such as day, week, month, day of the week, quarter, and week of the year.
It extracts field categories from column names using regular expressions.
It generates additional features based on correlation analysis and combines them with the original data.
It applies word embedding using Word2Vec on a combined categorical feature.
It calculates and incorporates target encoding based on certain categorical features.
It drops unnecessary columns and performs one-hot encoding on remaining categorical features.
It drops certain variable features.
The function returns the featurized data.

Scaling:
The code defines a scaling() function that scales the numerical features of the data.
It separates continuous features from other non-numerical features.
It applies StandardScaler to scale the continuous features.
It combines the scaled continuous features with non-numerical features.
The function returns the scaled data.

Model Training and Evaluation:
The code trains and evaluates logistic regression and XGBoost models on the preprocessed, featurized and scaled data.
It performs hyperparameter tuning using GridSearchCV for logistic regression and RandomizedSearchCV for XGBoost.
It evaluates the models using accuracy and AUC score.
The best models are saved using pickle.
The homesite_prediction() function predicts the target variable for a given input.
The homesite_evaluation() function evaluates the models on test data and calculates the AUC score and accuracy.
The code generates submission files for logistic regression and XGBoost models.