import os
from model import predict,load_model

def check_model_exist():
    assert os.path.exists("model.pkl")

def check_predictiion():
    pred=predict([8.5,40.0,6.1,1.2,320,2.55,37.4,-120.33])
    assert isinstance(pred,float)
