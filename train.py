import numpy as np
from model import predict, loss, gradient, normalize

#loading data

data = np.loadtxt("Data/apartments.csv", delimiter=",", skiprows=1)
sizes_raw = data[:,0]
prices = data[:,1]

#normalize sized

sizes, size_Mean, size_std = normalize(sizes_raw)

#initialize
w = 0.0
b = 0.0
learning_rate = 0.01
iterations = 1000  #how many times the process will be held

for i in range(iterations):
    predictions = predict(sizes, w, b)
    current_loss = loss(predictions, prices)
    dw, db = gradient(sizes, predictions, prices)
    
    w = w - learning_rate * dw
    b = b - learning_rate * db

    if i % 100 == 0:
        print(f"iteration {i}, loss: {current_loss}")
    
print("final w:", w, "final b:", b)