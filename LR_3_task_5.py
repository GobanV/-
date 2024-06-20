import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

plt.scatter(X, y, color='blue', label='Дані')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Випадкові дані (Варіант 1)')
plt.legend()
plt.show()

lin_reg = LinearRegression()
lin_reg.fit(X, y)

plt.scatter(X, y, color='blue', label='Дані')
plt.plot(X, lin_reg.predict(X), color='red', label='Лінійна регресія')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Лінійна регресія (Варіант 1)')
plt.legend()
plt.show()

degree = 10
poly_reg = PolynomialFeatures(degree=degree)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

plt.scatter(X, y, color='blue', label='Дані')
plt.plot(X, lin_reg_2.predict(X_poly), color='green', label=f'Поліноміальна регресія (ступінь {degree})')
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'Поліноміальна регресія (Варіант 1, ступінь {degree})')
plt.legend()
plt.show()

mse = mean_squared_error(y, lin_reg_2.predict(X_poly))
print("Середньо-квадратична помилка (MSE) поліноміальної регресії:", mse)
