
import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model('my_first_api')

# Define predict function
# 'Unnamed: 0': 4990, 'colour': 'White', 'year': 2002, 'age': 15, 'sex': 'Male', 'employed': 'Yes', 'citizen': 'Yes', 'checks': 2})
@app.post('/predict')
def predict(colour, age, sex, employed, citizen, checks):
    data = pd.DataFrame([[colour, age, sex, employed, citizen, checks]])
    data.columns = ['colour', 'age', 'sex', 'employed', 'citizen', 'checks']
    predictions = predict_model(model, data=data) 
    value_predic = predictions["prediction_label"].iloc[0]
    if (value_predic > 0.5):
        return f'Released Yes { value_predic }'
    else:
        return f'Released No { value_predic }'

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8002)