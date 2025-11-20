import numpy as np
import pickle


def load_model():
    with open("model.pkl","rb") as f:
        model=pickle.load(f)
    return model

def predict(sample):
    value=np.array(sample).reshape(1,-1)
    model=load_model()
    return float(model.predict(value)[0])