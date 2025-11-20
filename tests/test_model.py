import os
from models import predict,load_model

def test_check_model_exist():
    assert os.path.exists("model.pkl")

def test_check_predictiion():
    pred=predict([8.5,40.0,6.1,1.2,320,2.55,37.4,-120.33])
    assert isinstance(pred,float)
