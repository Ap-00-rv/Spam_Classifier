import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

model = joblib.load('model.pkl') 
vectorizer = joblib.load('vectorizer.pkl') 

def check_spam(message):
    try:
        input_data = vectorizer.transform([message])
        prediction = model.predict(input_data)
        return prediction[0] == 1  #
    except Exception as e:
        raise ValueError(f"Error in check_spam function: {e}")

@app.route('/receive', methods=['POST'])
def receive_message():
    try:
        data = request.get_json()
        message = data.get('message')
        
        is_spam = check_spam(message)
        
        return jsonify({'is_spam': is_spam})
    except Exception as e:
        app.logger.error(f"Error processing message: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)
