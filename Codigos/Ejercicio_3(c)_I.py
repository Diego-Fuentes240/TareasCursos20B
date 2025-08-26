import sympy as sp
import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Parámetros principales ----------------
grado_max = 15          # grado máximo 
mp.mp.dps = 80          # precision para integraciones numericas con mpmath

# ---------------- Definicion simbolica de la funcion h(x) ----------------
x = sp.symbols('x')
h_simbolica = sp.sin(3 * x * (1 - x**2))   # h(x) = sin(3 x (1-x^2))

# Versiones numericas para integrar y graficar
h_mpmath = sp.lambdify(x, h_simbolica, 'mpmath') 
h_numpy = sp.lambdify(x, h_simbolica, 'numpy')    

# ---------------- 1) Serie de Maclaurin (Taylor en 0) ----------------
serie_taylor_sim = sp.series(h_simbolica, x, 0, grado_max + 1).removeO()
serie_taylor_sim = sp.expand(serie_taylor_sim)  # expresion en base monomios

# ---------------- 2) Coeficientes de Legendre (proyeccion L2 en [-1,1]) ----------------
def polinomio_legendre_sim(k: int):

    return sp.legendre(k, x)

def polinomio_legendre_mpmath(k: int):
    
    return sp.lambdify(x, sp.legendre(k, x), 'mpmath')

coef_legendre = []  # lista de coeficientes c_k (numéricos, mpmath)
for k in range(0, grado_max + 1):
    Pk_m = polinomio_legendre_mpmath(k)
    integral = mp.quad(lambda t: h_mpmath(t) * Pk_m(t), [-1, 1])  # integral numerica
    ck = (2*k + 1) / 2 * integral
    coef_legendre.append(ck)

# Construir la aproximacion por Legendre truncada como expresion simbolica
aprox_legendre_sim = sum([sp.N(coef_legendre[k]) * polinomio_legendre_sim(k) for k in range(grado_max + 1)])
aprox_legendre_sim = sp.expand(aprox_legendre_sim)

# ---------------- 3) Coeficientes en base de monomios (como floats) ----------------
def coeficientes_monomios_float(polinomio_sim: sp.Expr, maxgrado: int):
   
    p_expand = sp.expand(polinomio_sim)
    lista = []
    for d in range(maxgrado + 1):
        c = sp.N(p_expand.coeff(x, d))
        try:
            lista.append(float(c))
        except Exception:
            lista.append(0.0)
    return lista

coef_taylor = coeficientes_monomios_float(serie_taylor_sim, grado_max)
coef_legendre_monomios = coeficientes_monomios_float(aprox_legendre_sim, grado_max)

# ---------------- 4) Buscar el primer grado donde difieren las truncadas ----------------
primer_grado_diferencia = None
tolerancia = 1e-9
for d in range(0, grado_max + 1):
    # Taylor truncado a grado d
    taylor_truncado_sim = sum([coef_taylor[k] * x**k for k in range(d + 1)])
    # Legendre usando solo P_0 .. P_d (grado <= d)
    legendre_truncado_sim = sum([sp.N(coef_legendre[k]) * polinomio_legendre_sim(k) for k in range(d + 1)])
    legendre_truncado_sim = sp.expand(legendre_truncado_sim)
    tc = coeficientes_monomios_float(taylor_truncado_sim, d)
    lc = coeficientes_monomios_float(legendre_truncado_sim, d)
    iguales = True
    for k in range(d + 1):
        if abs(tc[k] - lc[k]) > tolerancia:
            iguales = False
            break
    if not iguales:
        primer_grado_diferencia = d
        break

# ---------------- 5) Mostrar resultados ----------------
print("\n=== h(x) = sin(3 x (1 - x^2)) ===")
print("Grado maximo (N) =", grado_max, "\n")

print("Serie de Maclaurin (Taylor truncada hasta grado N):")
# Para evitar problemas de codificacion usamos pretty en modo ASCII
print(sp.pretty(serie_taylor_sim, use_unicode=False))

print("\nAproximacion por polinomios de Legendre (proyeccion L2, truncada hasta grado N):")
print(sp.pretty(aprox_legendre_sim, use_unicode=False))

print("\nCoeficientes monomiales (grado, coef_Taylor, coef_Legendre_en_monomios):")
print("{:>5s}  {:>20s}  {:>22s}".format("grado", "coef_Taylor", "coef_Legendre_monomio"))
for d in range(grado_max + 1):
    print(f"{d:5d}  {coef_taylor[d]:20.12e}  {coef_legendre_monomios[d]:22.12e}")

print("\nPrimer grado en que las expansiones truncadas difieren:", primer_grado_diferencia)
print("\nObservaciones:")
print(" - h(x) es impar: por eso los coeficientes en grados pares son (casi) cero.")
print(" - La serie de Taylor es una expansion local en x=0; la proyeccion en Legendre")
print("   minimiza el error en norma L2 en [-1,1]. Por eso los coeficientes difieren.\n")

# ---------------- 6) Graficas: comparar y mostrar errores ----------------
xx = np.linspace(-1, 1, 800)

# valores originales
y_original = h_numpy(xx)

# polinomios truncados completos hasta grado_max (en monomios)
taylor_gradoN_sim = sum([coef_taylor[k] * x**k for k in range(grado_max + 1)])
legendre_gradoN_sim = aprox_legendre_sim

taylor_func_numpy = sp.lambdify(x, taylor_gradoN_sim, 'numpy')
legendre_func_numpy = sp.lambdify(x, legendre_gradoN_sim, 'numpy')

y_taylor = taylor_func_numpy(xx)
y_legendre = legendre_func_numpy(xx)

# Grafica 1: funcion original y aproximaciones
plt.figure(figsize=(9,6))
plt.plot(xx, y_original, label='h(x) original', linewidth=2)
plt.plot(xx, y_taylor, '--', label=f'Taylor truncado (grado <= {grado_max})')
plt.plot(xx, y_legendre, ':', label=f'Proyeccion en Legendre (grado <= {grado_max})')
plt.axhline(0, color='gray', linewidth=0.6)
plt.legend()
plt.title('Comparacion: h(x) vs Taylor vs Legendre (truncadas)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.tight_layout()
plt.show()

# Grafica 2: error absoluto (escala log) en [-1,1]
error_taylor = np.abs(y_original - y_taylor)
error_legendre = np.abs(y_original - y_legendre)

plt.figure(figsize=(9,6))
plt.plot(xx, error_taylor, label=f'|h - Taylor_{grado_max}|')
plt.plot(xx, error_legendre, label=f'|h - Legendre_{grado_max}|')
plt.yscale('log')
plt.legend()
plt.title('Error absoluto (escala log) en [-1,1]')
plt.xlabel('x')
plt.ylabel('error (escala log)')
plt.grid(True, which='both')
plt.tight_layout()
plt.show()

# ---------------- Fin ----------------
