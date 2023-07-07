from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import os
import re
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# The following function performs data preprocessing based on the findings from the Exploratory Data Analysis section.
def preprocess(data):
    """
        Preprocesses the input data by performing various data cleaning and imputation steps.

        Parameters:
        - data (DataFrame): Input data to be preprocessed.

        Returns:
        - preprocessed_data (DataFrame): Preprocessed data with cleaned and imputed values.
    """
    # "Field10" has only integers and stored as text with comma seprators. Remove the comma separators to read it as integers.
    data['Field10'] = data['Field10'].apply(lambda comma: str(comma).replace(',',''))
    # While preprocessing the data, we noticed a null value in the "test" data that is filled with a space. The following
    # code is used to replace any spaces in the "PropertyField37" feature with the most frequently occurring value.
    data['PropertyField37'] = data['PropertyField37'].str.strip().replace('',data['PropertyField37'].value_counts().idxmax())
    # The following section of code is used to fill null values in the features with the most frequently occurring values.
    data['PropertyField38'] = data['PropertyField38'].fillna(data['PropertyField38'].value_counts().idxmax())
    data['PersonalField7'] = data['PersonalField7'].fillna(data['PersonalField7'].value_counts().idxmax())
    data['PropertyField36'] = data['PropertyField36'].fillna(data['PropertyField36'].value_counts().idxmax())
    data['PropertyField3'] = data['PropertyField3'].fillna(data['PropertyField3'].value_counts().idxmax())
    data['PropertyField32'] = data['PropertyField32'].fillna(data['PropertyField32'].value_counts().idxmax())
    data['PropertyField34'] = data['PropertyField34'].fillna(data['PropertyField34'].value_counts().idxmax())
    data['PropertyField4'] = data['PropertyField4'].fillna(data['PropertyField4'].value_counts().idxmax())
    # The following code performs mean value imputation.
    data['PersonalField84'] = data['PersonalField84'].fillna(data['PersonalField84'].mean())
    return data

# The below function does featurization on the train and test datasets.
def featurize(data):
    """
        Perform feature engineering on the input data.

        Parameters:
        - data (DataFrame): The input data to be featurized.
        - corr_50 (DataFrame): A DataFrame containing features correlated with the class label.
        - corr_features_25 (DataFrame): A DataFrame containing features correlated with each other.

        Returns:
        - DataFrame: The featurized data.
    """
    # The below section of code generate 6 features from the "Original_Quote_Date" feature.
    data['Day'] = data['Original_Quote_Date'].apply(lambda dt: datetime.strptime(dt,"%Y-%m-%d").day)
    data['Week'] = data['Original_Quote_Date'].apply(lambda dt: datetime.strptime(dt, "%Y-%m-%d").strftime("%U")).astype('int64')
    data['Month'] = data['Original_Quote_Date'].apply(lambda dt: datetime.strptime(dt,"%Y-%m-%d").month)
    data['day_of_week'] = pd.to_datetime(data['Original_Quote_Date']).dt.weekday
    data['quarter'] = pd.to_datetime(data['Original_Quote_Date']).dt.quarter
    # The "isocalendar" function provides a week-wise representation of dates and allows us to obtain the week of the year from it.
    data['week_of_year'] = pd.to_datetime(data['Original_Quote_Date']).dt.isocalendar().week.astype('int64')

    # The following section of code groups features based on the feature names and generate new features by finding aggregates.
    # The regular expression pattern given below matches a capitalized character followed by a lowercase word,
    # followed by any uppercase letter or digit.
    pattern = r"([A-Z][a-z]+)[A-Z0-9]"
    column_names = ' '.join(data.columns)
    # Apply the regex pattern on the string created by combining all the feature names.
    matches = re.findall(pattern, column_names)
    # Apply "set" to remove the duplicates.
    field_categories = list(set(matches))
    # Remove "Quote" to skip the "Original_Quote_Date" which is a unique column name.
    field_categories.remove('Quote')
    # Generate new features with each categories of features and initialize with "0".
    for i in field_categories:
        data[i] = 0
    # Loop over all the features and sum up values for the numerical features in same category.
    for i in data.columns:
        for j in field_categories:
            if (data[i].dtype=='int64') or (data[i].dtype=='float64'):
                if data[i].isnull().sum() == 0:
                    if str(i).startswith(j):
                        data[j].fillna(0, inplace=True)
                        data[j] += data[i]

    # The list "corr_50" contains the top 50 features that exhibit the highest correlation with the class label.
    # Iterate through each of the top 50 important features and create new features using mathematical functions such as
    # logarithm, square root, and inverse.
    corr_50 = ['PropertyField29', 'PropertyField35', 'Field9', 'CoverageField11B', 'GeographicField20A',\
                'CoverageField11A', 'GeographicField20B', 'PersonalField82', 'Field8', 'GeographicField17B',\
                'PersonalField26', 'GeographicField44A', 'GeographicField44B', 'PersonalField81', 'GeographicField38B',\
                'PersonalField83', 'PersonalField25', 'SalesField6', 'PersonalField80', 'PropertyField8', 'SalesField3',\
                'PropertyField1B', 'PersonalField22', 'PersonalField24', 'PropertyField1A', 'PropertyField15',\
                'GeographicField2A', 'GeographicField2B', 'PersonalField79', 'GeographicField38A', 'GeographicField41B',\
                'GeographicField36B', 'PersonalField23', 'PersonalField1', 'CoverageField5A', 'GeographicField36A',\
                'GeographicField39B', 'GeographicField46A', 'GeographicField35B', 'GeographicField22B', 'GeographicField18B',\
                'GeographicField46B', 'PersonalField11', 'PropertyField16B', 'GeographicField19A', 'GeographicField30B',\
                'GeographicField39A', 'PersonalField4B', 'PersonalField4A', 'GeographicField41A']
    for i in corr_50:
        if data[i].nunique() > 4:
            data[i + '_log'] = np.log(data[i])
            # The logarithm of "0" is equal to negative infinity. The following code will replace any infinity values with "0".
            data.loc[np.isinf(data[i + '_log']), i + '_log'] = 0
            # The logarithm of any negative values will return a NaN value. The following code will replace any NaN values with "0".
            data.loc[np.isnan(data[i + '_log']), i + '_log'] = 0
            data[i + '_sqrt'] = np.sqrt(data[i])
            data.loc[np.isnan(data[i + '_sqrt']), i + '_sqrt'] = 0
            data[i + '_invrs'] = data[i]
            # The inverse of "0" is undefined (or infinity). The code below only calculates the inverse for non-zero values.
            data.loc[data[i + '_invrs'] != 0, i + '_invrs'] = 1 / data[i]

    # The list "corr_25_feat" consists of 25 sets of features that exhibit high correlation with each other.
    # These sets of features are selected from the previously mentioned "corr_50" dataset. Let's generate new features
    # by combining these sets using addition, subtraction, multiplication, and division operations.
    corr_25_feat = [('GeographicField2A','GeographicField2B'),('PersonalField4B','PersonalField4A'),('GeographicField44B','GeographicField44A'),
                    ('PropertyField1A','PropertyField1B'),('GeographicField41B','GeographicField41A'),('GeographicField46B','GeographicField46A'),
                    ('PersonalField26','PersonalField25'),('PersonalField25','PersonalField24'),('PersonalField81','PersonalField80'),
                    ('PersonalField82','PersonalField81'),('GeographicField38A','GeographicField38B'),('GeographicField36A','GeographicField36B'),
                    ('GeographicField20A','Field9'),('PersonalField23','PersonalField22'),('PersonalField79','PersonalField80'),
                    ('PersonalField83','PersonalField79'),('PersonalField23','PersonalField24'),('CoverageField11A','CoverageField11B'),
                    ('PersonalField22','PersonalField24'),('GeographicField39A','GeographicField39B'),('PersonalField80','PersonalField83'),
                    ('PersonalField81','PersonalField83'),('PersonalField25','PersonalField22'),('PersonalField24','PersonalField26'),
                    ('PersonalField82','PersonalField80')]
    for i in corr_25_feat:
        data[i[0]+'_'+i[1]+'_add'] = data[i[0]] + data[i[1]]
        data[i[0]+'_'+i[1]+'_subt'] = data[i[0]] - data[i[1]]
        data[i[0]+'_'+i[1]+'_mul'] = data[i[0]] * data[i[1]]
        # To prevent divide-by-zero errors, assign the new feature with the values from the first feature. Then, divide
        # the second feature by the first feature, but only for non-zero values in the first feature.
        data[i[0]+'_'+i[1]+'_div'] = data[i[0]]
        data.loc[data[i[0]+'_'+i[1]+'_div']!=0, i[0]+'_'+i[1]+'_div'] = data[i[1]]/data[i[0]]

    # The below section of code is to concatenate all the categorical text features and generate a new feature using
    # the "Word2Vec" word embedding method.
    # Generate a list of indexes for the categorical features
    cat_features = data.dtypes[(data.dtypes!='int64')&(data.dtypes!='float64')].index
    # Drop the feature "Original_Quote_Date" as it is not a text feature.
    cat_features = cat_features.drop(['Original_Quote_Date'])
    # Concatenate the values in the categorical features for each data point while converting them to lowercase.
    data['combined_cat_features'] = data[cat_features].apply(lambda x: ''.join(x.dropna().astype(str).str.lower()), axis=1)
    # Create a list to store all the unique text values, which will be used as input for the Word2Vec model.
    texts = data['combined_cat_features'].tolist()
    # Convert all the texts in to each individual lists.
    sentences = [text.split() for text in texts]
    # Train the "Word 2 Vec" model with the sentence list created.
    model = Word2Vec(sentences,vector_size=128,min_count=1)
    # Obtrain word ID's from the "Word 2 Vec" model which will be used to match with the train/test dataset and the "Word 2 Vec" model outputs.
    word_ids = list(model.wv.index_to_key)
    # Obtrain word embeddings from the "Word 2 Vec" model. These are the vectors representing each texts in the sentence.
    word_embeddings = model.wv.vectors
    # Create a dataframe with "word_ids" and the corresponding "word_embeddings" to lookup from the train/test datasets.
    # The "Word 2 Vec" model returns 128 dimensional vectors as per the "vector_size" specified. Sum it up to get a single vector representation.
    embedding_df = pd.DataFrame({'word_ids':word_ids,'word_embeddings':[sum(i) for i in word_embeddings]})
    # Merge the "word_embeddings" to the train/test datasets.
    data = data.merge(embedding_df, left_on="combined_cat_features", right_on="word_ids", how="left")
    # Replace any values in the train/test dataframe with "0" that do not have a corresponding lookup in the "word_embeddings" table.
    data.loc[data['combined_cat_features']!=data['word_ids'],['word_embeddings']] = 0
    # Remove the text feature created by combining categorical features to obtain the word embeddings.
    data.drop(['combined_cat_features','word_ids'], axis=1, inplace=True)

    # The following code segment converts categorical features to numerical features for those features that have 12 or more unique values.
    # Calculate the frequency of each unique value in below features relative to the total number of data points.
    # Assign these frequencies to new variables. Update the respective features with the corresponding frequency values.
    CoverageField9_new = data.groupby('CoverageField9')['QuoteNumber'].count() / len(data)
    data['CoverageField9'] = data['CoverageField9'].apply(lambda cv: CoverageField9_new[cv])
    PersonalField16_new = data.groupby('PersonalField16')['QuoteNumber'].count() / len(data)
    data['PersonalField16'] = data['PersonalField16'].apply(lambda cv: PersonalField16_new[cv])
    PersonalField17_new = data.groupby('PersonalField17')['QuoteNumber'].count() / len(data)
    data['PersonalField17'] = data['PersonalField17'].apply(lambda cv: PersonalField17_new[cv])
    PersonalField18_new = data.groupby('PersonalField18')['QuoteNumber'].count() / len(data)
    data['PersonalField18'] = data['PersonalField18'].apply(lambda cv: PersonalField18_new[cv])
    PersonalField19_new = data.groupby('PersonalField19')['QuoteNumber'].count() / len(data)
    data['PersonalField19'] = data['PersonalField19'].apply(lambda cv: PersonalField19_new[cv])
    PropertyField7_new = data.groupby('PropertyField7')['QuoteNumber'].count() / len(data)
    data['PropertyField7'] = data['PropertyField7'].apply(lambda cv: PropertyField7_new[cv])

    # Remove the "QuoteNumber" feature, as it is not specific to individual users.
    # Additionally, drop the "Original_Quote_Date" feature, as we have already created date-related features.
    data.drop(['QuoteNumber','Original_Quote_Date'],axis=1, inplace=True)
    # Create a list of the names of categorical features that have 8 or fewer unique elements.
    categorical_feats = ['Field6','Field12','CoverageField8','SalesField7','PersonalField7','PropertyField3',\
                        'PropertyField4','PropertyField5','PropertyField14','PropertyField28','PropertyField30',\
                        'PropertyField31','PropertyField32','PropertyField33','PropertyField34','PropertyField36',\
                        'PropertyField37','PropertyField38','GeographicField63','GeographicField64']
    # Create one-hot encoded vectors for the categorical features mentioned above.
    encoded_data = pd.get_dummies(data[categorical_feats])
    # Remove the categorical features for which we have created one-hot encoded vectors.
    data = pd.concat([data.drop(categorical_feats,axis=1),encoded_data], axis=1)

    # Remove the features from the dataset that have a variance less than 0.001. These features are listed in the variable 'var_features'.
    var_features = ['Field9', 'PersonalField8', 'PersonalField64', 'PersonalField65', 'PersonalField66', 'PersonalField69', 'PropertyField6', 'PropertyField9', 'PropertyField20', 'PropertyField29', 'GeographicField10A']
    data.drop(var_features, axis=1, inplace=True)

    col_names = set(data.columns)
    if os.path.isfile('Pickles/col_names.pkl'):
        with open('Pickles/col_names.pkl','rb') as f:
            col_names_train = pickle.load(f)
        diff_cols = col_names.symmetric_difference(col_names_train)
        diff_cols.remove('QuoteConversion_Flag')
        for i in diff_cols:
            data[i] = 0
    return data

def scaling(data):
    """
        Perform scaling on the input data.

        Parameters:
        - data (DataFrame): The input data to be scaled.

        Returns:
        - numpy.ndarray: The scaled and combined data.
    """
    # The following code section performs standard scaling on the train and test datasets.
    # For "test" data, we load the parameters of standard scaler and the continuous features from pickle files.
    # Load the scaler and continuous features to perform scaling on the test data.
    if os.path.isfile('Pickles/scaler.pkl') & os.path.isfile('Pickles/continuous_features.pkl'):
        with open('Pickles/scaler.pkl','rb') as f:
            scaler = pickle.load(f)
        with open('Pickles/continuous_features.pkl','rb') as f:
            continuous_features = pickle.load(f)
        data_num = data[continuous_features]
        data_non_num = data.drop(continuous_features,axis=1)
        data_num_scaled = scaler.transform(data_num)
    else:
        # For test data, exist the function if there is pickle file found for the scaler and the continuous features.
        return
    # Concatenate the scaled and non-scaled datasets to create the final dataset for modeling.
    data_combined = np.concatenate((data_num_scaled,np.array(data_non_num)),axis=1)
    # Return the scaled data.
    return data_combined

# Define route for the home page
@app.route('/')
def home():
    return render_template('Homesite Demo_3.html')

# Define route for form submission
@app.route('/predict', methods=['POST'])

def predict():
    # Retrieve input values from the form
    inputs_numerical = ['GeographicField','PersonalField','PropertyField','SalesField','CoverageField','Field']
    inputs_categorical = ['Original_Quote_Date','Field6','Field12','CoverageField8','CoverageField9','SalesField7',\
                          'PersonalField7','PersonalField16','PersonalField17','PersonalField18','PersonalField19',\
                          'PropertyField3','PropertyField4','PropertyField5','PropertyField7','PropertyField14',\
                          'PropertyField28','PropertyField30','PropertyField31','PropertyField32','PropertyField33',\
                          'PropertyField34','PropertyField36','PropertyField37','PropertyField38','GeographicField63',\
                          'GeographicField64']
    user_inputs_num = {}
    for i in inputs_numerical:
        user_inputs_num[i] = request.form[i]
    user_inputs_cat = {}
    for i in inputs_categorical:
        user_inputs_cat[i] = request.form[i]

    with open('Pickles/means.pkl','rb') as f:
        means = pickle.load(f)
    with open('Pickles/devs.pkl','rb') as f:
        devs = pickle.load(f)
    
    user_inputs = {}
    for i in means.index:
        for j in user_inputs_num:
            user_inputs[i] = [(np.random.normal(loc=means[i], scale=devs[i])*0.75)+(int(user_inputs_num[j])*0.25)]
    user_inputs = {**user_inputs,**user_inputs_cat,**{'QuoteNumber':np.random.randint(1,100000)}}
    user_inputs = pd.DataFrame(user_inputs)
    
    # Preprocess the data using the function "preprocess".
    user_inputs = preprocess(user_inputs)
    # Featurize the data using the function "featurize".
    user_inputs = featurize(user_inputs)
    # Do standard scaling on the data using the function "scaling".
    user_inputs = scaling(user_inputs)
    # Call the machine learning model and pass the input values for prediction
    with open('Pickles/xgb_model.pkl','rb') as f:
        xgb_model = pickle.load(f)
    # Predict the "QuoteConversion_Flag" using the pretrained model.
    prediction = xgb_model.predict(user_inputs)
    # Prepare the output message
    if prediction:
        output_message = "accept"
    else:
        output_message = "decline"
    # Render the output page with the prediction result
    return render_template('Homesite Demo_3.html', prediction=output_message)

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=8080)

#if __name__ == '__main__':
    # Run the Flask app
#    app.run(debug=True)