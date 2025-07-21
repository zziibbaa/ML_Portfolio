from flask import Flask , jsonify , request
from joblib import dump , load
import pandas as pd

## create flask app
app=Flask(__name__)

# create API routing call
@app.route('/predict' , methods=['POST']) # http://localhost:5000/predict
def predict():

    # get json Request
    feat_data = request.json
    # convert json request to pandas DataFrame
    df = pd.DataFrame(feat_data)
    # match columns name
    df = df.reindex(columns=col_names)
    # get prediction
    prediction = list(model.predict(df))
    # return json version of prediction
    return jsonify({'prediction': str(prediction)})

if __name__ == '__main__':

    # load model & feature columns
    model=load('final_model.pkl')
    col_names=load('column_name.pkl')


    app.run(debug=True)