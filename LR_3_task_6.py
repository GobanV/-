import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve

m = 100
X = np.linspace(-3, 3, m)
y = 2 * np.sin(X) + np.random.uniform(-0.6, 0.6, m)
X_reshaped = X.reshape(-1, 1)


def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=5, scoring='neg_mean_squared_error',
                                                            train_sizes=np.linspace(0.1, 1.0, 10))
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Тренувальні дані")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Перевірочні дані")

    plt.xlabel("Розмір тренувальної вибірки")
    plt.ylabel("Середньо-квадратична помилка")
    plt.title("Крива навчання")
    plt.legend(loc="best")
    plt.grid()
    plt.show()


lin_reg = LinearRegression()
plot_learning_curve(lin_reg, X_reshaped, y)

degree = 10
poly_reg = PolynomialFeatures(degree=degree)
X_poly = poly_reg.fit_transform(X_reshaped)
lin_reg_2 = LinearRegression()
plot_learning_curve(lin_reg_2, X_poly, y)
