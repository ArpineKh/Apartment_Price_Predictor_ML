import numpy as np
import matplotlib.pyplot as plt
from model import predict, normalize

data = np.loadtxt("Data/apartments.csv", delimiter=",", skiprows=1)
sizes_raw = data[:,0]
prices = data[:,1]

sizes, size_mean, size_std = normalize(sizes_raw)

w = 50677.49114086321
b = 127266.666452481

predictions = predict(sizes, w, b)

plt.scatter(sizes_raw, prices, color="blue", label="Actual Data")
plt.plot(sizes_raw, predictions, color="red", label="Fitted Line")
plt.xlabel("Size (m²)")
plt.ylabel("Price")
plt.legend()
plt.show()
