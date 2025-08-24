import sympy as sp
from sympy import Matrix, symbols


# Removed unused symbol definitions

v1 = Matrix([1, 1])
v2 = Matrix([2, 3])
v3 = Matrix([4, 3])

g = (v1 + v2 + v3) / 3

print(g) 