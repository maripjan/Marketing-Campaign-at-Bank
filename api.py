from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Create Flask app
app = Flask(__name__)

# Load the trained model from the file
model_pipeline = joblib.load('ml_pipeline.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.get_json()
    
    # Convert JSON data to pandas DataFrame
    X_new = pd.DataFrame(data['X']) # Assuming that key there is called "X"
    
    # Normalize and make predictions
    predictions = model_pipeline.predict(X_new)
    
    # Send predictions back as JSON
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
