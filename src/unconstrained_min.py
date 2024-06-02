import numpy as np
import matplotlib

def conv_check( f_conv, gap, obj_tol, param_tol):
    return f_conv < obj_tol or gap < param_tol

def gradient_descent(f, x0, steps, obj_tol, param_tol, max_iter):
    x=x0
    flag_conv = False
    x_prev = np.inf
    y_prev = np.inf
    val_y = []

    for i in range(max_iter):
        #init_instance
        [y_pres, gradient, hes] = f(x)
        val_y.append({'ind': x, 'val': y_pres})
        f_conv = np.abs(y_prev - y_pres)
        gap = np.linalg.norm(x - x_prev)
        x_prev = x
        x = x - steps * gradient

        if conv_check(f_conv, gap, obj_tol, param_tol):
            flag_conv = True
            break
        y_prev = y_pres
    
    print(f"Final Iteration: x: {x}, f(x): {y_pres}, Success: {flag_conv}")
    return {'res' : flag_conv, 'record': val_y}

def wolfe_searching_steps(x_target, p_target, c_1, a_init, f, gradient, c_tack):
    check_obj = False
    a_update = a_init
    count = 0
    #loop to update.
    while check_obj is False:
        #set vals, lhs/rhs
        [rhs, _, _] = f(x_target)
        [lhs, _, _] = f(x_target + a_update * p_target)
        rhs = rhs + c_1 * (gradient @ p_target) * a_update 
        #update
        a_update = c_tack * a_update
        check_obj = (-0.0001 < rhs - lhs)
        count += 1
        #check qual to return
        if count > 500:
            return a_init
    #otherwise recent alpha returned
    return a_update


def min_lin(f, x0, steps, obj_tol, param_tol, max_iter, select_dir, step_from=1.0,
                c_1=0.01, c_tack=0.5):
    x=x0
    flag_conv = False
    x_prev = np.inf
    y_prev = np.inf
    val_y = []

    for i in range(max_iter):
        #init_instance
        [y_pres, gradient, hes] = f(x)
        val_y.append({'ind':  x, 'val': y_pres})
        f_conv = abs(y_prev - y_pres)        
        gap = np.linalg.norm(x - x_prev)
   
        #conv_check before direction
        if conv_check(f_conv, gap, obj_tol, param_tol):
            flag_conv = True
            break
             
        #see direction we need
        if select_dir == 'gradient_descent':
            p_target = - steps * gradient
        if select_dir == 'newton':
            p_target = np.linalg.solve(hes, -gradient)
        
        #allocate updated alpha using wolfe condition
        if i == 0:
            a_update = step_from
        else:
            a_update = wolfe_searching_steps(p_prev, p_prev, c_1, step_from, f, gradient, c_tack)
            
        #update
        x_prev = x
        x = x + a_update * p_target
        y_prev = y_pres
        p_prev = p_target
    print(f"Final Iteration: x: {x}, f(x): {y_pres}, Success: {flag_conv}")
    return {'res': flag_conv, 'record': val_y}

