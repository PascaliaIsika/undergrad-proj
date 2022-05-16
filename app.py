from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
import pickle

#create an application
app = Flask(__name__)

#load machine learning model
model = joblib.load("model.pkl")

# bind home function to URL
@app.route('/')
def home():
    return render_template("index2.html")

#bind predict function to URL
@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    input_features = [int(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = [ "id","age","gender","height","weight","ap_hi","ap_lo","cholesterol","gluc","smoke","alco","active"]
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)

    if output == 1:
        res_val = "cardiovascular diseases"
    else:
        res_val = "no cardiovascular diseases"
         
    
       
    return render_template('index2.html',prediction_text='patient has {}'.format(res_val))

    
    
    

        


         

     
# Running the app
if __name__ == '__main__':
    app.run()
