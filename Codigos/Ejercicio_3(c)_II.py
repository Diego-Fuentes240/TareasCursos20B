import sympy as sp
import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Parámetros ----------------
grado_max = 15            # grado máximo 
mp.mp.dps = 80            # precisión para integraciones numéricas con mpmath
tol = 1e-9                # tolerancia para comparar coeficientes

# ---------------- Definición de la función h(x) ----------------
x = sp.symbols('x')
h_sim = sp.sin(3*x) * (1 - x**2)    # h(x) = sin(3x) * (1 - x^2)
h_mpm = sp.lambdify(x, h_sim, 'mpmath')  # para integrar con mpmath
h_np = sp.lambdify(x, h_sim, 'numpy')    # para evaluar con numpy (graficar)

# ---------------- 1) Serie de Maclaurin (base de monomios) ----------------
serie_taylor_sim = sp.series(h_sim, x, 0, grado_max + 1).removeO()
serie_taylor_sim = sp.expand(serie_taylor_sim)   # expresión en monomios

# ---------------- 2) Base de Chebyshev de 2ª especie (U_n) y producto interno ------------
def U_sim(k: int):
    return sp.chebyshevu(k, x)

def U_mpm(k: int):
    return sp.lambdify(x, sp.chebyshevu(k, x), 'mpmath')

def peso_para_integracion(t):
    
    return mp.sqrt(1 - t**2)


def producto_interno_con_peso(f_mpm, g_mpm):
    integrando = lambda t: f_mpm(t) * g_mpm(t) * peso_para_integracion(t)
    return mp.quad(integrando, [-1, 1])

# ---------------- 3) Coeficientes de proyección en U_n (chebyshev 2ª especie) ------------

coef_cheby = []
norma_U = []   # guardamos <U_k, U_k> para cada k

for k in range(grado_max + 1):
    Uk_m = U_mpm(k)
    # <h, U_k>
    numerador = producto_interno_con_peso(h_mpm, Uk_m)
    # <U_k, U_k>
    denominador = producto_interno_con_peso(Uk_m, Uk_m)
    coefk = numerador / denominador
    coef_cheby.append(coefk)
    norma_U.append(denominador)

# Construir aproximación por Chebyshev truncada (suma_{k=0}^N c_k U_k(x))
aprox_cheby_sim = sum([sp.N(coef_cheby[k]) * U_sim(k) for k in range(grado_max + 1)])
aprox_cheby_sim = sp.expand(aprox_cheby_sim)

# ---------------- 4) Convertir aproximación Chebyshev a coeficientes en monomios ------
def coef_monomios_float(poly_sim: sp.Expr, maxgrado: int):
    p = sp.expand(poly_sim)
    lista = []
    for d in range(maxgrado + 1):
        c = sp.N(p.coeff(x, d))
        try:
            lista.append(float(c))
        except Exception:
            lista.append(0.0)
    return lista

coef_taylor = coef_monomios_float(serie_taylor_sim, grado_max)
coef_cheby_monomios = coef_monomios_float(aprox_cheby_sim, grado_max)

# ---------------- 5) Encontrar el primer grado en que difieren las truncadas ---------
primer_grado_diferencia = None
for d in range(0, grado_max + 1):
    # Taylor truncada a grado d
    taylor_trunc_sim = sum([coef_taylor[k] * x**k for k in range(d + 1)])
    # Chebyshev truncada considerando solo U_0..U_d (grado <= d)
    cheby_trunc_sim = sum([sp.N(coef_cheby[k]) * U_sim(k) for k in range(d + 1)])
    cheby_trunc_sim = sp.expand(cheby_trunc_sim)
    tc = coef_monomios_float(taylor_trunc_sim, d)
    lc = coef_monomios_float(cheby_trunc_sim, d)
    iguales = True
    for k in range(d + 1):
        if abs(tc[k] - lc[k]) > tol:
            iguales = False
            break
    if not iguales:
        primer_grado_diferencia = d
        break

# ---------------- 6) Mostrar resultados ----------------
print("\n=== Inciso II: h(x) = sin(3x) * (1 - x^2) ===")
print("Grado máximo usado (N) =", grado_max, "\n")

print("Serie de Maclaurin (Taylor truncada hasta grado N):")
print(sp.pretty(serie_taylor_sim, use_unicode=False))

print("\nAproximación por polinomios de Chebyshev de 2ª especie (U_n), truncada hasta grado N:")
# mostramos la aproximación en la base U_n:
# por claridad imprimimos la suma c_k U_k
print("Coeficientes c_k (proyección) en la base U_k (valores numéricos):")
for k in range(grado_max + 1):
    print(f" k={k:2d}  c_k = {float(coef_cheby[k]): .12e}   (||U_k||^2 = {float(norma_U[k]): .12e})")

print("\nAproximación por Chebyshev expresada en la base de monomios (expandida):")
print(sp.pretty(aprox_cheby_sim, use_unicode=False))

print("\nTabla de coeficientes monomiales (grado, coef_Taylor, coef_Cheby_en_monomios):")
print("{:>5s}  {:>20s}  {:>22s}".format("grado", "coef_Taylor", "coef_Cheby_monomio"))
for d in range(grado_max + 1):
    print(f"{d:5d}  {coef_taylor[d]:20.12e}  {coef_cheby_monomios[d]:22.12e}")

print("\nPrimer grado en que las expansiones truncadas difieren:", primer_grado_diferencia)
print("\nObservaciones:")
print(" - La base de monomios (Taylor) es una expansión local en x=0;")
print(" - La proyección en U_n minimiza el error en la norma L2 con peso sqrt(1-x^2) en [-1,1];")
print(" - Por ello, incluso coeficientes de bajo grado pueden diferir entre ambas expansiones.\n")

# ---------------- 7) Graficar: función original y aproximaciones ----------------
xx = np.linspace(-1, 1, 1000)
y_orig = h_np(xx)

# polinomios truncados grado <= grado_max en monomios
pol_taylor_N = sum([coef_taylor[k] * x**k for k in range(grado_max + 1)])
pol_cheby_N = aprox_cheby_sim

f_taylor = sp.lambdify(x, pol_taylor_N, 'numpy')
f_cheby = sp.lambdify(x, pol_cheby_N, 'numpy')

y_taylor = f_taylor(xx)
y_cheby = f_cheby(xx)

plt.figure(figsize=(10,6))
plt.plot(xx, y_orig, label='h(x) original', linewidth=2)
plt.plot(xx, y_taylor, '--', label=f'Taylor (grado ≤ {grado_max})')
plt.plot(xx, y_cheby, ':', label=f'Proyección en U_n (grado ≤ {grado_max})')
plt.legend()
plt.title('Comparación: h(x) vs Taylor vs Chebyshev (U_n) — inciso II')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.tight_layout()
plt.show()

# Graficar errores absolutos (escala log)
err_taylor = np.abs(y_orig - y_taylor)
err_cheby = np.abs(y_orig - y_cheby)

plt.figure(figsize=(10,6))
plt.plot(xx, err_taylor, label=f'|h - Taylor_{grado_max}|')
plt.plot(xx, err_cheby, label=f'|h - Chebyshev(U)_{grado_max}|')
plt.yscale('log')
plt.legend()
plt.title('Error absoluto (escala log) en [-1,1] — inciso II')
plt.xlabel('x')
plt.ylabel('error (log)')
plt.grid(True, which='both')
plt.tight_layout()
plt.show()

# ---------------- Fin ----------------
