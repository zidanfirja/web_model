from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

# Load trained model from model.pkl
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        data = request.form
        input_features = np.array([[int(data['Sex']), 
                                    float(data['Age']), 
                                    float(data['Passenger_Fare']),
                                    int(data['Passenger_Class']),
                                    int(data['Port_of_Embarkation'])]])
        
        # Predict using the loaded model
        prediction = model.predict(input_features)[0]
        probability = model.predict_proba(input_features)[0][1]  # Probability of survival

        # Response message
        if prediction == 1:
            result = {
                "prediction": "SELAMAT",
                "message": f"Penumpang kemungkinan besar akan selamat dengan probabilitas {probability:.2f}."
            }
        else:
            result = {
                "prediction": "TIDAK SELAMAT ",
                "message": f"Penumpang tersebut kemungkinan besar tidak akan selamat dengan probabilitas {1 - probability:.2f}."
            }

        return render_template('result.html', result=result)
    except Exception as e:
        return jsonify({'error': 'Invalid input or server error', 'details': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
