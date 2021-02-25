import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [int(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
   
    features_name = ['age','workclass','education_level','education-num','marital_status',
                     'occupation','relationship','race','sex','capital-gain','capital-loss',
                     'hours-per-week','native_country']   
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)       
    if output == '<=50K':
        res_val = "may not donate to charity."
    else:
        res_val = "donates to charity."
    return render_template('index.html', prediction_text='User {}'.format(res_val))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)




if __name__ == "__main__":
    app.run(debug=True)
