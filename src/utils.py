import matplotlib.pyplot as plt
import numpy as np

def plot_contour(f, path, title):
    X, Y = np.meshgrid( np.linspace(-3, 3), np.linspace(-3, 3))
    Z = np.array([f(np.array(p))[0] for p in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)
    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=5)
    plt.title(title)
    for v in path:
        plt.plot(v['ind'][0], v['ind'][1],'o-')
    plt.savefig("Contour" + title + '.png')
    plt.show()

def plot_obj_iter(path, title):
    values = [x["val"] for x in path]
    plt.figure(figsize=(10, 8))
    plt.plot(range(len(path)), values, 'o-')
    plt.xlabel('Iterations')
    plt.ylabel('Objective Function')
    plt.grid(True)
    plt.title(title)
    plt.savefig("OI" + title + '.png')
    plt.show()
