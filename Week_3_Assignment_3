import math
import numpy as np
import random
from scipy.optimize import minimize

def gen_model(num_rivers):
    s = 10 * np.random.rand(num_rivers)
    v = 2 * np.random.rand(num_rivers)
    v_p = 2 * random.random()
    T = 10 * num_rivers / 2 * random.random() #10 and 2 are normalization factors
    return s, v, v_p, T

def objective(s, v, v_p, a):
    # s * v / v_p / math.cos(math.radians(a)) + math.tan(math.radians(a))
    res = s * v / v_p
    rad_a = map(math.radians, a)
    res /= list(map(math.cos, rad_a))
    res += list(map(math.tan, rad_a))
    return res

def constraint(s, v_p, a):
    # s / v_p / math.cos(math.radians(a))
    res = s / v_p
    rad_a = map(math.radians, a)
    res /= list(map(math.cos, rad_a))
    return res

def optimize(s, v, v_p, T, a_init):
    # optimize
    bound = (0.0, 90.0)
    bounds = (bound, bound, bound, bound)
    cons = {'type': 'ineq', 'fun': constraint(s, v_p, a_init)}
    #con2 = {'type': 'eq', 'fun': constraint2}
    #cons = ([con1, con2])
    solution = minimize(objective, a_init, method='SLSQP', bounds=bounds, constraints=cons)
    return solution.x

def run():
    num_rivers = 10
    s, v, v_p, T = gen_model(num_rivers)
    a_init = np.random.rand(num_rivers)
    return optimize(s, v, v_p, T, a_init)

if __name__ == '__main__':	# 跑.py的时候，跑main下面的；被导入当模块时，main下面不跑，其他当函数调
    run()
