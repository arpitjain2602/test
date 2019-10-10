import pandas as pd
from flask import Flask, jsonify, request
import pickle
import os
import inspect


'''
The structure of the code follows:
1. Load pickled model
2. Name flask app
3. Create a route that receives JSON inputs, uses the trained model to make a prediction, 
and returns that prediction in a JSON format, which can be accessed through the API endpoint.


-
Inside the route, I converted the JSON data to a pandas dataframe object because 
I found that this works with most (not all!) types of models that you will want to use to make a 
prediction. You can choose to convert the inputs using your preferred method as long as it works 
with the .predict() method for your model. The order of inputs must match the order of columns in the 
dataframe that you used to train your model otherwise you will get an error when you try to make a prediction. 
If the inputs you are receiving are not in the correct order you can easily reorder them after you create the dataframe.

-
The takeaway here is that you need to convert the JSON data 
that you get from the request to a data structure that the 
model can use to make a prediction. However you get there is up to you.
'''

# load model
model_pkl = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),'model.pkl')
model = pickle.load(open(model_pkl,'rb'))

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

def predict():
    # get data
    data = request.get_json(force=True)

    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)

    # predictions
    result = model.predict(data_df)

    # send back to browser
    output = {'results': int(result[0])}

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)