# Apartment_Price_Predictor_ML

## Goal
Predict apartment price from size using **linear regression**, trained from scratch with **gradient descent** — no ML libraries, just NumPy.

## Folder structure
```
apartment_price_predictor/
├── data/
│   └── apartments.csv
├── model.py     ← the math: predict(), loss(), gradient(), normalize()
├── train.py     ← the process: load data, train, print results
└── plot.py      ← visualize the fit
```

---

## The core idea in one line
**guess → measure error → compute slope of error → step downhill → repeat**



## 1. `predict()` — the model itself

**Formula:** `price = w · size + b`

- `w` (weight) — how much price changes per extra m² (the slope)
- `b` (bias) — baseline price when size = 0 (the intercept)

**Matrix form** (for multiple features later): `predicted = X @ w + b`, where `X` is a matrix of apartments × features, and `w` is a vector of weights — the dot product computed for every apartment at once.

```python
def predict(size, w, b):
    return w * size + b
```

## 2. `loss()` — how wrong the model is

**Formula (Mean Squared Error):**

`loss = mean( (predicted − actual)² )`

Why squared: keeps all errors positive (no cancellation) and punishes big mistakes more than small ones.

```python
def loss(predictions, actual):
    return np.mean((predictions - actual) ** 2)
```

## 3. `gradient()` — which direction reduces the loss

Derived by hand using the **chain rule** on `loss = (w·size + b − actual)²`:

**`∂loss/∂w = mean( 2 · (predicted − actual) · size )`**
**`∂loss/∂b = mean( 2 · (predicted − actual) )`**

The sign of the gradient tells you the direction that *increases* loss — so training always moves the *opposite* way.

```python
def gradient(size, predicted, actual):
    error = predicted - actual
    dw = np.mean(2 * error * size)
    db = np.mean(2 * error)
    return dw, db
```

## 4. `normalize()` — putting features on the same scale

**Formula:** `normalized_x = (x − mean) / standard_deviation`

- `x − mean` centers the data around 0
- `÷ std` rescales so "1" means "one typical deviation from average," regardless of original units

**Why it matters:** since `size` multiplies directly into the gradient (`dw = mean(2·error·size)`), features with very different raw scales produce wildly different-sized gradients — forcing an impossible tradeoff on a single shared learning rate (too big for one feature, too slow for another). Normalizing makes gradients comparable in size, so one learning rate works well for all weights, and gradient descent converges directly instead of zigzagging.

```python
def normalize(x):
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / std, mean, std
```

---

## 5. The training loop — `train.py`

```python
import numpy as np
from model import predict, loss, gradient, normalize

data = np.loadtxt("data/apartments.csv", delimiter=",", skiprows=1)
sizes_raw = data[:, 0]
prices = data[:, 1]

sizes, size_mean, size_std = normalize(sizes_raw)

w = 0.0
b = 0.0
learning_rate = 0.01
iterations = 1000

for i in range(iterations):
    predictions = predict(sizes, w, b)
    current_loss = loss(predictions, prices)
    dw, db = gradient(sizes, predictions, prices)

    w = w - learning_rate * dw
    b = b - learning_rate * db

    if i % 100 == 0:
        print(f"iteration {i}, loss: {current_loss}")

print("final w:", w, "final b:", b)
```

**Update rule:** `w = w − learning_rate × gradient` — this single formula self-corrects direction automatically, since subtracting a negative gradient is the same as adding.

---

## Concept → subject map

| Piece | Subject |
|---|---|
| `w·size + b`, dot products, `X @ w` | Linear algebra |
| MSE, mean, standard deviation, normalization | Statistics |
| Chain rule, derivatives, `∂loss/∂w`, `∂loss/∂b` | Calculus |
| Loops, functions, NumPy arrays, unpacking, indentation | Python |

## Still ahead
- **Step 8**: plot the fitted line against real data points (`plot.py`)
- Extending to multiple features (rooms, floor, age) → true matrix form `X @ w`
- Part 2 of the original plan: a Naive Bayes spam classifier, to directly drill probability/Bayes' theorem
