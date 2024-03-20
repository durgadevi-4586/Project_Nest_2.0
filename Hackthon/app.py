from flask import Flask,render_template,request
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
app=Flask(__name__)
#loading dataset
insurance=pd.read_csv("data.csv")
#converting categorical data into numerical data
encoder = LabelEncoder()
insurance["sex"] = encoder.fit_transform(insurance["sex"])
insurance["smoker"] = encoder.fit_transform(insurance["smoker"])
insurance["region"] = encoder.fit_transform(insurance["region"])
#taking x and y values
x = insurance.drop(columns='charges',axis = 1)
y = insurance['charges']
#splitting the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=20, random_state=42)
#model fitting
regression= LinearRegression()
regression.fit(x_train,y_train)
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/result',methods=['GET', 'POST'])
def predict():
    if request.method=='POST':
        # Get the input values from the form
        age = int(request.form['age'])
        sex = str(request.form['sex'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = str(request.form['smoker'])
        region = str(request.form['region'])

        # Encode categorical variables
        sex_encoded = encoder.fit_transform([sex])[0]
        smoker_encoded = encoder.fit_transform([smoker])[0]
        region_encoded = encoder.fit_transform([region])[0]

        # Prepare input data for prediction
        new_data = np.array([age, sex_encoded, bmi, children, smoker_encoded, region_encoded]).reshape(1, -1)

        # Make prediction
        prediction = regression.predict(new_data)
        print(prediction)

        return render_template('result.html', Prediction=prediction[0])



if __name__=='__main__':
    app.run(debug=True)