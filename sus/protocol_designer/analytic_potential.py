import numpy as np

def test_potenial(var, params):
    x, y, z = var
    a, b, c = params

    return a*x**2 + b*y**2 + c*z**2

def test_force(var, params):
    x, y, z = var
    a, b, c = params

    return -2*a*x, -2*b*y, -2*c*z


