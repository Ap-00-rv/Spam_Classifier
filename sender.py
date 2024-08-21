from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

RECEIVER_URL = 'http://localhost:5001/receive'

@app.route('/send', methods=['POST'])
def send_message():
    data = request.json
    if not data or 'message' not in data:
        app.logger.error('Invalid request payload')
        return jsonify({"status": "Invalid request payload"}), 400

    message = data.get('message')

    try:
        response = requests.post(RECEIVER_URL, json={'message': message})
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        app.logger.error(f'Request failed: {e}')
        return jsonify({"status": "Request failed", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
