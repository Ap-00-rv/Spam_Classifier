from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Path to the CSV file and log file
csv_file_path = 'spam.csv'
log_file_path = 'spam_log.txt'

# Load the dataset with 'latin1' encoding
df = pd.read_csv(csv_file_path, encoding='latin1')
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
df.rename(columns={'v1': 'Category', 'v2': 'text'}, inplace=True)
df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})  # Convert 'ham'/'spam' to 0/1

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    message = data['message']
    
    # Preprocess and transform the message
    transformed_message = vectorizer.transform([message]).toarray()
    
    # Predict using the loaded model
    prediction = model.predict(transformed_message)[0]
    
    # Return the prediction
    return jsonify({'prediction': 'spam' if prediction == 1 else 'ham'})

@app.route('/test_all', methods=['GET'])
def test_all():
    log_content = ''
    try:
        # Check if DataFrame is loaded and not empty
        if df.empty:
            return jsonify({'error': 'DataFrame is empty or failed to load.'})

        # Filter DataFrame to include only spam messages
        spam_df = df[df['Category'] == 1]
        
        if spam_df.empty:
            return jsonify({'status': 'completed', 'message': 'No spam messages found in the dataset.'})
        
        # Print the number of spam messages found
        print(f"Found {len(spam_df)} spam messages.")
        
        # Iterate over each row in the filtered DataFrame
        for index, row in spam_df.iterrows():
            message = row['text']
            
            # Preprocess and transform the message
            transformed_message = vectorizer.transform([message]).toarray()
            
            # Predict using the loaded model
            prediction = model.predict(transformed_message)[0]
            result = 'spam' if prediction == 1 else 'ham'
            
            # Append message and prediction to log content
            log_content += f'Message: {message}\nPrediction: {result}\n\n'
        
        # Write all log content to the file with latin1 encoding
        with open(log_file_path, 'w', encoding='latin1') as log_file:
            log_file.write(log_content)
        
        # Check if log file has been written
        if os.path.getsize(log_file_path) == 0:
            return jsonify({'status': 'completed', 'message': 'Log file is empty.'})
        
        return jsonify({'status': 'completed', 'log_file': log_file_path})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
