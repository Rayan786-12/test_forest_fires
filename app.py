# from flask import Flask,request,jsonify,render_template
# import pickle
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import Ridge
# ridge_model=pickle.load(open('ridge.pkl','rb'))
# standard_scaler=pickle.load(open('scaler.pk1','rb'))


# application=Flask(__name__)
# app=application
# @app.route('/')
# def hello_world():
#     return render_template('index.html',methods=['GET','POST'])

# @app.route('/predictdata',methods=['POST'])
# def predict_datapoint():
#     if request.method=='POST':
#         Temperature=float(request.form.get('Temperature'))
#         RH=float(request.form.get('RH'))
#         Ws=float(request.form.get('Ws'))
#         Rain= float(request.form.get('Rain'))
#         FFMC= float(request.form.get('FFMC'))
#         DMC=float(request.form.get("DMC"))
#         ISI=float(request.form.get("ISI"))
#         Classes=float(request.form.get("Classes"))
#         Region=float(request.form.get('Region'))

#         input_data = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
#         result = ridge_model.predict(input_data)
#         return render_template('home.html', results=result[0])
#     else:
#         return render_template('home.html',results=None)
# if __name__=='__main__':
#     app.run(host='0.0.0.0')

from flask import Flask, request, render_template
import os
import pickle
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

MODEL_PATH = 'ridge.pkl'
SCALER_PATH = 'scaler.pk1'

def train_and_save_model():
    # ❗ Dummy data — replace with your real dataset if available
    X_train = np.array([
        [25, 60, 4, 0.2, 88, 18, 2.5, 1, 3],
        [20, 45, 5, 0.1, 85, 15, 3.0, 2, 2],
        [30, 55, 3, 0.0, 90, 20, 2.0, 1, 1]
    ])
    y_train = np.array([80, 75, 78])  # target variable

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = Ridge()
    model.fit(X_scaled, y_train)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    return model, scaler

# Load model and scaler with version mismatch fallback
try:
    with open(MODEL_PATH, 'rb') as f:
        ridge_model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        standard_scaler = pickle.load(f)
except Exception as e:
    print(f"⚠️ Model loading failed: {e}\n🔁 Retraining model...")
    ridge_model, standard_scaler = train_and_save_model()
    print("✅ Model and scaler retrained and saved.")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    try:
        # Get data from form
        data = [
            float(request.form['Temperature']),
            float(request.form['RH']),
            float(request.form['Ws']),
            float(request.form['Rain']),
            float(request.form['FFMC']),
            float(request.form['DMC']),
            float(request.form['ISI']),
            float(request.form['Classes']),
            float(request.form['Region'])
        ]

        # Scale and predict
        input_data = np.array([data])
        scaled_data = standard_scaler.transform(input_data)
        prediction = ridge_model.predict(scaled_data)

        return render_template('home.html', results=round(prediction[0], 2))

    except Exception as e:
        return f"❌ Error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
