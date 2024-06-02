import numpy as np
import math as math


def quadratic_1(x):
    Q = np.array([[1, 0], [0, 1]])
    f_val = (x.T) @ Q @ x
    grad = Q @ x
    hes = Q 
    return f_val, grad, hes

def quadratic_2(x):
    Q = np.array([[1, 0], [0, 100]])
    f_val = (x.T) @ Q @ x
    grad =  Q @ x
    hes =  Q 
    return f_val, grad, hes

def quadratic_3(x):
    Q = np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]]).T @ np.array([[100, 0], [0, 1]]) @ np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])
    f_val = (x.T) @ Q @ x
    grad =  Q @ x
    hes = Q
    return f_val, grad, hes


def rosenbrock(x):
    f_val = 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    grad = np.array([-400 * x[0] *(x[1] - x[0]**2) - 2*(1 - x[0]), 200 *(x[1] - x[0]**2)])
    hes = np.array([[1200 * x[0] **2 - 400* x[1] + 2, -400 * x[0]], [-400 *x[0], 200]])
    return f_val, grad, hes

def linear_function(x):
    a = np.array([2, -1])
    f_val = a.T @ x
    grad = a
    hes = np.zeros((2, 2))
    return f_val, grad, hes

def corner_tri(x):
    e_1 = np.exp(x[0] + 3 * x[1] - 0.1)
    e_2 = np.exp(x[0] - 3 * x[1] - 0.1)
    e_3 = np.exp(-x[0] - 0.1)

    f_val = e_1 + e_2 + e_3
    grad = np.array([e_1+e_2-e_3, 3*e_1 - 3*e_2])
    hes = np.array([ [e_1 + e_2 + e_3, 3 * e_1 - 3 * e_2],[3 * e_1 - 3 * e_2, 9 * e_1 + 9 * e_2]])

    return f_val, grad, hes

