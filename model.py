import numpy as np


def predict(size, w, b):
    return w*size+b;

def loss(predictions, actual):
    return np.mean((predictions-actual)**2);

def gradient(size, predicted, actual):
    error = predicted - actual;
    dw = np.mean(2 * error * size); #derivative of loss, if we set x as w. I.e. changing w to see the behavior of loss
    db = np.mean(2 * error); #same for b
    
    return dw, db

def normalize(x):
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean)/std, mean, std
    



print("prediction is" , predict(80, 2000, 10000)) 
predictions = np.array([80000, 110000, 150000])
actual = np.array([52000, 75000, 102000])
print(loss(predictions, actual))

sizes = np.array([60])
actual = np.array([90000])
predictions = predict(sizes, 1500, 5000)  
dw, db = gradient(sizes, predictions, actual)
print(dw, db)

sizes = np.array([35, 42, 50, 58, 63, 70, 78, 85, 92, 100, 110, 120, 130, 145, 160])
normalized, mean, std = normalize(sizes)
print("normalized sizes" , normalized, "mean", mean, "std", std)
