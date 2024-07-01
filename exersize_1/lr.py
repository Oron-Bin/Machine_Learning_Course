import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate training data
rng = np.random.RandomState(42)
x = np.linspace(0, 10, num=2000).reshape(-1,1)
y = 3.2 * x + rng.normal(scale=x / 2)

# Fit linear model
model = LinearRegression()
model.fit(x, y)

# Predict values for the x range
y_predicted = model.predict(x)

# Plot
plt.figure()
plt.plot(x, y, 'o', alpha=0.5, markersize=1, label = 'Data')
plt.plot(x, y_predicted, '-k', label = 'Predicted')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()