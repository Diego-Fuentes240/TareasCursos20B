import sympy as sp
from sympy import Matrix 
x = sp.symbols("x")    

#(a). Muestre que la base dada no es ortogonal

#Solo hace falta demostrar que un subconjunto no es ortogonal para demostrar que todo el conjunto no lo es
base = [Matrix([1]), Matrix([x]), Matrix([x**2]), Matrix([x**3]), Matrix([x**4]), Matrix([x**5]), Matrix([x**6]),
Matrix([x**7]), Matrix([x**8]), Matrix([x**9])]

def producto_interno(f, g):
    return sp.integrate(f[0] * g[0], (x, -1, 1))

ortogonales = True
for i, vi in enumerate(base):
    for j, vj in enumerate(base):
        if j > i:
            ip = producto_interno(vi, vj)
            if ip != 0:
                ortogonales = False
                print(f"El producto interno de {vi[0]} y {vj[0]} es: {ip}")
if ortogonales:       
    print("El conjunto de vectores es ortogonal")
else:
    print("El conjunto de vectores no es ortogonal")

#----------------------------------------------------------------------------------

#(b). Use el procedimiento de Gram-Schmidt para encontrar una base ortogonal
#Ahora ortogonalizamos la base con Gram-Schmidt
#Con esto se encuentran los 10 primeros polinomios de Legendre

def ortogonalizar(base):
    ortogonales = []
    for v in base:
        w = v
        for u in ortogonales:
            w -= (producto_interno(v, u)/producto_interno(u, u))*u
        ortogonales.append(sp.simplify(w))
    return ortogonales

base_ortogonal = ortogonalizar(base)

print("\nNueva base de vectores ortogonalizados:")
for i, p in enumerate(base_ortogonal):
    print(f"{p[0]}")

#----------------------------------------------------------------------------------

# (c). Ortogonalizacion para un producto interno diferente
def producto_interno_peso(h, m):
    return sp.integrate(h[0] * m[0] * sp.sqrt(1 - x**2), (x, -1, 1))

def ortogonalizar_2(base):
    ortogonales_2 = []
    for v in base:
        w = v
        for u in ortogonales_2:
            w -= (producto_interno_peso(v, u)/producto_interno_peso(u, u))*u
        ortogonales_2.append(sp.simplify(w))
    return ortogonales_2

base_ortogonal_peso = ortogonalizar_2(base)

print("Base de vectores ortogonalizados para un producto interno diferente:")
for i, p in enumerate(base_ortogonal_peso):
    print(f"{p[0]}")

#----------------------------------------------------------------------------------

#(d). Suponga la función h(x) = sen(3x)(1 − x2):

# I.
import sympy as sp
from sympy import Matrix
from sympy.plotting import plot

# 1. Definimos la variable y la funcion a expandir
x = sp.symbols("x")
h_x = sp.sin(3*x) * (1 - x**2)
limite_inferior, limite_superior = -1, 1

# 2. Definimos la base de monomios y el grado de la expansion
grado_expansion = 3
base_monomios = [x**i for i in range(grado_expansion + 1)]

# 3. Definimos el producto interno
def producto_interno_monomios(f, g):
    return sp.integrate(f * g, (x, limite_inferior, limite_superior))

# 4. Construimos la matriz Gramian (A)
n = grado_expansion + 1
A = Matrix(n, n, lambda i, j: producto_interno_monomios(base_monomios[i], base_monomios[j]))

# 5. Construimos el vector de terminos independientes (b)
b = Matrix(n, 1, lambda i, j: producto_interno_monomios(h_x, base_monomios[i]))

# 6. Resolvemos el sistema de ecuaciones para encontrar los coeficientes (c)
# c = A.inv() * b
coeficientes = A.LUsolve(b)

# 7. Construimos la expansion de la funcion h(x)
expansion_h = sum(coeficientes[i] * base_monomios[i] for i in range(n))
expansion_h_simplificada = sp.simplify(expansion_h)

# 8. Mostramos los resultados
print("La matriz Gramian (A) es:")
print(A)
print("\nEl vector de terminos independientes (b) es:")
print(b)
print("\nLos coeficientes de la expansion (c) son:")
print(coeficientes)
print("\nLa expansion de h(x) en la base de monomios es:")
print(expansion_h_simplificada)

plot(h_x, expansion_h_simplificada, (x, limite_inferior, limite_superior),
    title='Expansion de h(x) en base de monomios',
    legend=True,
    show=True)