#for f(x)=x^2; g(x)=x+1; y=f(g(x))
# Define the functions
def f(x):
    return x ** 2

def g(x):
    return x + 1

def chain_rule_derivative(x):
    # Compute the derivative of f(g(x)) using the chain rule
    df_dg = 2 * g(x)
    dg_dx = 1
    dy_dx = df_dg * dg_dx
    return dy_dx

# Test the chain rule implementation
x = 2
dy_dx = chain_rule_derivative(x)
print("dy_dx =", dy_dx)

import sympy as sp

# Define the symbols and functions
x = sp.Symbol('x')
f = x ** 2
g = x + 1

# Compute the derivative of f(g(x)) using the chain rule
dy_dx = sp.diff(f.subs(x, g), x)

# Evaluate the derivative at a specific point
dy_dx_value = dy_dx.subs(x, 2)
print("dy_dx =", dy_dx_value)

from scipy.misc import derivative

# Define the functions
def f(x):
    return x ** 2

def g(x):
    return x + 1

# Define the composition function
def composition(x):
    return f(g(x))

# Compute the derivative using the chain rule
x = 2
dy_dx = derivative(composition, x)

# Print the result
print("dy_dx =", dy_dx)

import torch

# Define the functions
def f(x):
    return x ** 2

def g(x):
    return x + 1

# Define the input variable
x = torch.tensor(2.0, requires_grad=True)

# Perform the computation
y = f(g(x))

# Compute the derivative using the chain rule
dy_dx = torch.autograd.grad(y, x)[0]

# Print the result
print("dy_dx =", dy_dx.item())

