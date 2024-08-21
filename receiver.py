import joblib
import numpy as np
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

# Load the pre-trained model and vectorizer
model = joblib.load('model.pkl')  # Adjust the path as necessary
vectorizer = joblib.load('vectorizer.pkl')  # Adjust the path as necessary

# Define the path to the log file
LOG_FILE_PATH = 'spam_results.log'

def check_spam(message):
    try:
        # Transform the message into the format the model expects
        input_data = vectorizer.transform([message])
        prediction = model.predict(input_data)
        return bool(prediction[0])  # Convert to boolean
    except Exception as e:
        raise ValueError(f"Error in check_spam function: {e}")

def log_result(message, is_spam):
    try:
        # Open the log file in append mode
        with open(LOG_FILE_PATH, 'a') as log_file:
            log_file.write(f"{datetime.now()}: Message: {message}\n")
            log_file.write(f"{datetime.now()}: is_spam: {is_spam}\n")
            log_file.write("\n")  # Add a newline for readability
    except Exception as e:
        app.logger.error(f"Error writing to log file: {e}")

@app.route('/receive', methods=['POST'])
def receive_message():
    try:
        data = request.get_json()
        message = data.get('message')
        
        # Process the message
        is_spam = check_spam(message)
        
        # Log the result
        log_result(message, is_spam)
        
        # Return a JSON response
        return jsonify({'is_spam': is_spam})
    except Exception as e:
        app.logger.error(f"Error processing message: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)
