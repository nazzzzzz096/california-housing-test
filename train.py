from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestClassifier


data=fetch_california_housing()
print("Feature names:", data.feature_names)
print("Data shape:", data.data.shape)
print("Target shape:", data.target.shape)
print("First row of data:\n", data.data[0])
print("First target value:", data.target[0])

x_train,x_test,y_train,y_test=train_test_split(data.data,data.target,random_state=42,test_size=0.2)

model=RandomForestClassifier(n_estimator=50)

model.fit(x_train,y_train)

with open("model.pkl","wb") as f:
    pickle.dump(model)

