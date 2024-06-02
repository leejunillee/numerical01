
import numpy as np
import sys,os,unittest
sys.path.append(os.path.abspath('src'))
from src.unconstrained_min import gradient_descent, min_lin
from src.utils import plot_contour, plot_obj_iter
from tests.examples import quadratic_1, quadratic_2, quadratic_3, rosenbrock, linear_function, corner_tri
 
class TestUnconstrained(unittest.TestCase):

    def test_quadratic(self):
        for func in [quadratic_1, quadratic_2, quadratic_3, linear_function, corner_tri]:
            x0 = np.array([1, 1])
            changes = 0.01
            param_tol = 10**-8
            obj_tol = 10**-12
            max_itr = 100
            for method in ['newton','gradient_descent']:
                print(str(func)+'with '+method)
                res = min_lin(func, x0, changes, obj_tol, param_tol, max_itr, method)
                plot_contour( func, res['record'], str(func).split(" ")[1]+'with '+method)
                plot_obj_iter(res['record'], str(func).split(" ")[1]+'with '+method)
    def test_rosenbrock(self):
            x0 = np.array([-1, 2])
            changes = 0.01
            param_tol = 10**-8
            obj_tol = 10**-12
            max_itr = 10000
            for method in ['newton','gradient_descent']:
                print("rosenbrock"+'with '+method)
                res = min_lin(rosenbrock, x0, changes, obj_tol, param_tol, max_itr, method, step_from=0.005)
                plot_contour(rosenbrock, res['record'], 'rosenbrock ' + method)
                plot_obj_iter(res['record'], 'rosenbrock ' + method)
if __name__ == '__main__':
    unittest.main()


'''

class TestUnconstrainedMin(unittest.TestCase):
    
    def test_quadratic_1(self):
        x0 = np.array([1, 1])
        obj_tol = 1e-12
        param_tol = 1e-8
        max_iter = 100
        c_1 = 0.01
        factor_backtracking = 0.5

        methods = {
            "Gradient Descent": gradient_descent,
            "Newton's Method": search_lin,
        }

        for method, minimize_func in methods.items():
            if (minimize_func == gradient_descent):
                result = minimize_func(quadratic_1, x0, 1, obj_tol, param_tol, max_iter)
            else:
                result = minimize_func(quadratic_1, x0, 1, obj_tol, param_tol, max_iter, method, c_1=c_1, factor_backtracking=factor_backtracking)
            plot_contour_iter(result['record'], method)
            plot_contour_iter(result['record'])


    def test_quadratic_2(self):
        x0 = np.array([1, 1])
        obj_tol = 1e-12
        param_tol = 1e-8
        max_iter = 100
        c_1 = 0.01
        factor_backtracking = 0.5

        methods = {
            "Gradient Descent": gradient_descent,
            "Newton's Method": search_lin,

        }

        for method, minimize_func in methods.items():
            if (minimize_func == gradient_descent):
                result = minimize_func(quadratic_2, x0, 1, obj_tol, param_tol, max_iter)
            else:
                result = minimize_func(quadratic_2, x0, 1, obj_tol, param_tol, max_iter, method, c_1=c_1, factor_backtracking=factor_backtracking)
            plot_contour_iter(result['record'], method)
            plot_obj_iter(result['record'])

    def test_quadratic_3(self):
        x0 = np.array([1, 1])
        obj_tol = 1e-12
        param_tol = 1e-8
        max_iter = 100
        c_1 = 0.01
        factor_backtracking = 0.5

        methods = {
            "Gradient Descent": gradient_descent,
            "Newton's Method": search_lin,

        }

        for method, minimize_func in methods.items():
            if (minimize_func == gradient_descent):
                result = minimize_func(quadratic_3, x0, 1, obj_tol, param_tol, max_iter)
            else:
                result = minimize_func(quadratic_3, x0, 1, obj_tol, param_tol, max_iter, method, c_1=c_1, factor_backtracking=factor_backtracking)
            plot_contour_iter(result['record'], method)
            plot_obj_iter(result['record'])

    def test_rosenbrock(self):
        x0 = np.array([-1, 2])
        obj_tol = 1e-12
        param_tol = 1e-8
        max_iter = 10000  # Adjusted for the Rosenbrock example
        c_1 = 0.01
        factor_backtracking = 0.5

        methods = {
            "Gradient Descent": gradient_descent,
            "Newton's Method": search_lin,

        }

        for method, minimize_func in methods.items():
            if (minimize_func == gradient_descent):
                result = minimize_func(rosenbrock, x0, 1, obj_tol, param_tol, max_iter)
            else:
                result = minimize_func(rosenbrock, x0, 1, obj_tol, param_tol, max_iter, method, c_1=c_1, factor_backtracking=factor_backtracking)
            plot_contour_iter(result['record'], method)
            plot_obj_iter(result['record'])


    def test_linear_function(self):
        x0 = np.array([1, 1])
        obj_tol = 1e-12
        param_tol = 1e-8
        max_iter = 100
        c_1 = 0.01
        factor_backtracking = 0.5

        methods = {
            "Gradient Descent": gradient_descent,
            "Newton's Method": search_lin,
        }

        for method, minimize_func in methods.items():
            if (minimize_func == gradient_descent):
                result = minimize_func(linear_function, x0, 1, obj_tol, param_tol, max_iter)
            else:
                result = minimize_func(linear_function, x0, 1, obj_tol, param_tol, max_iter, method, c_1=c_1, factor_backtracking=factor_backtracking)
            plot_contour_iter(result['record'], method)
            plot_obj_iter(result['record'])


    def test_corner_tri(self):
        x0 = np.array([1, 1])
        obj_tol = 1e-12
        param_tol = 1e-8
        max_iter = 100
        c_1 = 0.01
        factor_backtracking = 0.5

        methods = {
            "Gradient Descent": gradient_descent,
            "Newton's Method": search_lin,

        }

        for method, minimize_func in methods.items():
            if (minimize_func == gradient_descent):
                result = minimize_func(corner_tri, x0, 1, obj_tol, param_tol, max_iter)
            else:
                result = minimize_func(corner_tri, x0, 1, obj_tol, param_tol, max_iter, method, c_1=c_1, factor_backtracking=factor_backtracking)
            plot_contour_iter(result['record'], method)
            plot_obj_iter(result['record'])


if __name__ == '__main__':
    unittest.main()
'''