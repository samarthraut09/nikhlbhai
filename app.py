import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import json

app = Flask(__name__)
## Load the model
regmodel = pickle.load(open('reg_model.pkl',"rb"))
scalar=pickle.load(open('scaling.pkl','rb'))  # we are using it in the model

@app.route('/')
def home():
    return render_template('house.html')

# @app.route('/predict_api', methods = ['POST'])
# def predict_api():
#     data = request.json['data']  
    
#     # """
#     # request.json['data'] :
#     #  It indicates that  whenever we hit predict api then whatever input we will
#     #  put it into the json format which is capture inside the data key
#     # """
#     print(data)
#     print(list(data.values())) # data is in dictionary format and we have to provide it into list
#     print(np.array(list(data.values())).reshape(1,-1))
#     new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))

#     output  = regmodel.predict(new_data)
#     print(output[0])
#     return render_template('house.html', prediction_text = "The House Price Prediction is {}.format{output}")

    #return jsonify({'Result':f"House Price Prediction is: {output[0]}"})

@app.route('/predict', methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    output1 = round(output,2)
    # return jsonify({'Result':f"House Price Prediction is: {output[0]}"})
    return render_template('home.html', prediction_text = "The House Price Prediction is ${} Million".format(output1))





if __name__ == "__main__":
    app.run(debug=True)


