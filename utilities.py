import matplotlib.pyplot as plt
import numpy as np

def visualize_classifier(classifier, X, y, title='Classifier boundaries'):
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    mesh_step_size = 0.01
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size),
                                 np.arange(min_y, max_y, mesh_step_size))
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    output = output.reshape(x_vals.shape)
    plt.figure()
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='k', linewidth=1, marker='o')
    plt.title(title)
    plt.show()
