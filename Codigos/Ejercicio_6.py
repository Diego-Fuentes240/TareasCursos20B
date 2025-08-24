import sympy as sp

# Definimos la variable
x = sp.Symbol('x')

# Producto interno con el peso sqrt(1 - x^2)
def producto_interno(f, g):
    return sp.integrate(f*g*sp.sqrt(1-x**2), (x, -1, 1))

# Procedimiento de Gram–Schmidt
def ortogonalizar(base):
    ortogonales = []
    for v in base:
        w = v
        for u in ortogonales:
            w -= (producto_interno(v, u)/producto_interno(u, u))*u
        ortogonales.append(sp.simplify(w))
    return ortogonales

# Base inicial: {1, x, x^2, x^3, x^4}
base_inicial = [1, x, x**2, x**3, x**4]

# Aplicamos el proceso
base_ortogonal = ortogonalizar(base_inicial)

# Mostramos los resultados
for i, p in enumerate(base_ortogonal):
    print(f"Polinomio ortogonal U_{i}(x) = {sp.simplify(p)}")
