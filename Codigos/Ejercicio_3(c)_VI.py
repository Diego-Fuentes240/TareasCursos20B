
import sympy as sp
import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Parámetros ----------------
mp.mp.dps = 80               # precisión para integraciones numéricas
grado_max = 35               # grado máximo a evaluar 
puntos_grilla = 3000         # número de puntos para evaluar funciones en [-1,1]
tolerancia = 1e-12

# ---------------- Definición simbólica de la función h(x) ----------------
x = sp.symbols('x')
h_sim = sp.sin(3*x) * (1 - x**2)    # h(x)
h_mpm = sp.lambdify(x, h_sim, 'mpmath')
h_np = sp.lambdify(x, h_sim, 'numpy')

# Grill para evaluación y norma
xx = np.linspace(-1, 1, puntos_grilla)
y_exact = h_np(xx)
peso = np.sqrt(1 - xx**2)  # peso sqrt(1-x^2)

# ---------------- Funciones auxiliares para coeficientes ----------------

def coeficientes_taylor_hasta(N: int):
    """Devuelve coeficientes de la serie de Maclaurin (floats) hasta grado N."""
    serie = sp.series(h_sim, x, 0, N+1).removeO()
    serie = sp.expand(serie)
    coefs = []
    for k in range(N+1):
        c = sp.N(serie.coeff(x, k))
        try:
            coefs.append(float(c))
        except Exception:
            coefs.append(0.0)
    return coefs

def coeficientes_legendre_hasta(N: int):

    coef = []
    for k in range(N+1):
        Pk = sp.lambdify(x, sp.legendre(k, x), 'mpmath')
        integral = mp.quad(lambda t: h_mpm(t) * Pk(t), [-1, 1])
        ck = (2*k + 1) / 2 * integral
        coef.append(ck)
    return coef

def coeficientes_chebyshevU_hasta(N: int):
  
    coef = []
    for k in range(N+1):
        Uk = sp.lambdify(x, sp.chebyshevu(k, x), 'mpmath')
        numer = mp.quad(lambda t: h_mpm(t) * Uk(t) * mp.sqrt(1 - t**2), [-1, 1])
        denom = mp.quad(lambda t: Uk(t) * Uk(t) * mp.sqrt(1 - t**2), [-1, 1])
        # protección contra denom ~ 0 (no debería pasar para U_k)
        if abs(denom) < mp.mpf('1e-30'):
            ck = mp.mpf('0')
        else:
            ck = numer / denom
        coef.append(ck)
    return coef

# ---------------- Evaluadores: construir polinomio desde coeficientes ----------------

def evaluar_taylor_desde_coef(coef, puntos):
   
    y = np.zeros_like(puntos)
    for k, c in enumerate(coef):
        y = y + c * (puntos ** k)
    return y

def evaluar_legendre_desde_coef(coef, puntos):
   
    y = np.zeros_like(puntos)
    for k, c in enumerate(coef):
        Pk_np = sp.lambdify(x, sp.legendre(k, x), 'numpy')
        y = y + float(c) * Pk_np(puntos)
    return y

def evaluar_chebyU_desde_coef(coef, puntos):
   
    y = np.zeros_like(puntos)
    for k, c in enumerate(coef):
        Uk_np = sp.lambdify(x, sp.chebyshevu(k, x), 'numpy')
        y = y + float(c) * Uk_np(puntos)
    return y

# ---------------- Cálculo de coeficientes completos hasta grado_max ----------------
print("Calculando coeficientes hasta grado", grado_max, "(esto puede tardar unos segundos)...")
coef_taylor_full = coeficientes_taylor_hasta(grado_max)
coef_legendre_full = coeficientes_legendre_hasta(grado_max)
coef_chebyU_full = coeficientes_chebyshevU_hasta(grado_max)
print("Coeficientes calculados. Empezando evaluación de errores por grado...")

# ---------------- Bucle por grado para calcular errores ----------------
grados = list(range(1, grado_max + 1))

errores_sup_taylor = []
errores_sup_legendre = []
errores_sup_chebyU = []

errores_L2_taylor = []
errores_L2_legendre = []
errores_L2_chebyU = []

errores_L2peso_taylor = []
errores_L2peso_legendre = []
errores_L2peso_chebyU = []

for n in grados:
    # construir y evaluar polinomios truncados grado n
    ct = coef_taylor_full[:n+1]
    y_taylor = evaluar_taylor_desde_coef(ct, xx)

    cl = coef_legendre_full[:n+1]
    y_legendre = evaluar_legendre_desde_coef(cl, xx)

    cu = coef_chebyU_full[:n+1]
    y_chebyU = evaluar_chebyU_desde_coef(cu, xx)

    # normas
    # norma sup (infinito)
    errores_sup_taylor.append(np.max(np.abs(y_exact - y_taylor)))
    errores_sup_legendre.append(np.max(np.abs(y_exact - y_legendre)))
    errores_sup_chebyU.append(np.max(np.abs(y_exact - y_chebyU)))

    # norma L2 discreta (raíz de media cuadrática en la grilla)
    errores_L2_taylor.append(np.sqrt(np.mean((y_exact - y_taylor)**2)))
    errores_L2_legendre.append(np.sqrt(np.mean((y_exact - y_legendre)**2)))
    errores_L2_chebyU.append(np.sqrt(np.mean((y_exact - y_chebyU)**2)))

    # norma L2 con peso sqrt(1-x^2)
    errores_L2peso_taylor.append(np.sqrt(np.mean(((y_exact - y_taylor)**2) * peso)))
    errores_L2peso_legendre.append(np.sqrt(np.mean(((y_exact - y_legendre)**2) * peso)))
    errores_L2peso_chebyU.append(np.sqrt(np.mean(((y_exact - y_chebyU)**2) * peso)))

print("Cálculo de errores completado.")

# ---------------- Gráficas ----------------
plt.figure(figsize=(9,5))
plt.semilogy(grados, errores_sup_taylor, '-o', label='Taylor (||·||_∞)')
plt.semilogy(grados, errores_sup_legendre, '-o', label='Legendre (||·||_∞)')
plt.semilogy(grados, errores_sup_chebyU, '-o', label='Chebyshev U (||·||_∞)')
plt.xlabel('grado n')
plt.ylabel('error (norma suprema)')
plt.title('Error supremo vs grado (métodos: Taylor, Legendre, Chebyshev U)')
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,5))
plt.semilogy(grados, errores_L2_taylor, '-o', label='Taylor (L2 discreto)')
plt.semilogy(grados, errores_L2_legendre, '-o', label='Legendre (L2 discreto)')
plt.semilogy(grados, errores_L2_chebyU, '-o', label='Chebyshev U (L2 discreto)')
plt.xlabel('grado n')
plt.ylabel('error (L2 discreto)')
plt.title('Error L2 (discreto) vs grado')
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,5))
plt.semilogy(grados, errores_L2peso_taylor, '-o', label='Taylor (L2 con peso)')
plt.semilogy(grados, errores_L2peso_legendre, '-o', label='Legendre (L2 con peso)')
plt.semilogy(grados, errores_L2peso_chebyU, '-o', label='Chebyshev U (L2 con peso)')
plt.xlabel('grado n')
plt.ylabel('error (L2 con peso sqrt(1-x^2))')
plt.title('Error L2 con peso sqrt(1-x^2) vs grado (norma natural para U_n)')
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.show()

# ---------------- Tabla resumen para grados seleccionados ----------------
grados_referencia = [5, 10, 15, 25, min(35, grado_max)]
print("\nResumen (norma suprema) para grados de referencia:")
for n in grados_referencia:
    if n <= grado_max:
        idx = n - 1
        print(f" n={n:2d}: Taylor={errores_sup_taylor[idx]:.3e}, Legendre={errores_sup_legendre[idx]:.3e}, ChebyU={errores_sup_chebyU[idx]:.3e}")

# ---------------- Ajuste exponencial aproximado para Chebyshev U (norma sup) ---------------
# Ajustamos en rango intermedio para ver tasa de decaimiento: log(error) ≈ a + b n
import numpy as _np
nmin_fit, nmax_fit = 6, min(20, grado_max)
y_fit = _np.array(errores_sup_chebyU[nmin_fit-1:nmax_fit])  # slice correcta
n_fit = _np.arange(nmin_fit, nmax_fit+1)
mask = y_fit > 0
if mask.any():
    coef_fit = _np.polyfit(n_fit[mask], _np.log(y_fit[mask]), 1)
    pendiente, intercepto = coef_fit[0], coef_fit[1]
    print(f"\nAjuste exponencial aproximado (norma sup) para Chebyshev U en n={nmin_fit}..{nmax_fit}:")
    print(f" log(error) ≈ {pendiente:.3f} * n + {intercepto:.3f}")
    print(f" => error ≈ exp({intercepto:.3f}) * exp({pendiente:.3f} * n)")

print("\nAnálisis finalizado.")

