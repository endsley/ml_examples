import sympy as sp

# Define the variable
x = sp.symbols('x')

# Define the function
f = (2*x + 1)**2

# Compute the derivative
f_prime = sp.diff(f, x)

# Print the result
print("The derivative of f(x) is:", f_prime)
