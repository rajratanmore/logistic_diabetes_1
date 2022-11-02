
import pandas as pd 
import numpy as np
import pickle 
import config
import json 

class Diabetes():
    def __init__(self, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
        self.Glucose = Glucose
        self.BloodPressure =BloodPressure
        self.SkinThickness = SkinThickness
        self.Insulin = Insulin
        self.BMI = BMI
        self.DiabetesPedigreeFunction = DiabetesPedigreeFunction
        self.Age = Age 

    def load_model(self):
        # with open(config.JSON_FILE_PATH, 'r') as f:
        #     self.json_data =json.load(f)
        with open(config.MODEL_FILE_PATH, 'rb') as f:
            self.model= pickle.load(f)
    def prediction(self):
        self.load_model()

        array= np.zeros(7)

        array[0]= self.Glucose
        array[1] = self.BloodPressure
        array[2]=self.SkinThickness
        array[3]=self.Insulin
        array[4]=self.BMI
        array[5] = self.DiabetesPedigreeFunction
        array[6]= self.Age
        print(array)

        prediction = self.model.predict([array])[0]

        return(prediction)

if __name__ == '__main__' :
    Glucose =85.000
    BloodPressure =66.000
    SkinThickness =29.000
    Insulin =0.000
    BMI =26.600
    DiabetesPedigreeFunction = 0.351
    Age =31.000

    result = Diabetes(Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age )

    Final = result.prediction()

    print('Preiction is ::::::', Final)
