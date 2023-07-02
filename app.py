from flask import Flask, render_template, request
import random
# Initialize Flask app
app = Flask(__name__)

# Define route for the home page
@app.route('/')
def home():
    return render_template('Homesite Demo_3.html')

# Define route for form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input values from the form
    inputs = ['GeographicField','PersonalField','PropertyField','SalesField','CoverageField','Field',\
              'Original_Quote_Date','Field6','Field12','CoverageField8','CoverageField9','SalesField7',\
              'PersonalField7','PersonalField16','PersonalField17','PersonalField18','PersonalField19',\
              'PropertyField3','PropertyField4','PropertyField5','PropertyField7','PropertyField14',\
              'PropertyField28','PropertyField30','PropertyField31','PropertyField32','PropertyField33',\
              'PropertyField34','PropertyField36','PropertyField37','PropertyField38','GeographicField63',\
              'GeographicField64']
    user_input = {}
    for i in inputs:
        user_input[i] = request.form[i]

    # Call the machine learning model and pass the input values for prediction
    #prediction = your_model_function(geographic_field, personal_field, property_field, sales_field, coverage_field, field, quote_date)
    # Prepare the output message
    prediction = 0
    if prediction:
        output_message = "accept"
    else:
        output_message = "decline"
    # Render the output page with the prediction result
    return render_template('Homesite Demo_3.html', prediction=user_input)

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)