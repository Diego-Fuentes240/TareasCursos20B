import sympy as sp
import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt

# ------- Parámetros -------
grado_N = 15        # grado de truncado para las aproximaciones
mp.mp.dps = 50      # precisión para integraciones numéricas
intervalo = (-1.0, 1.0)
puntos = 1000       # número de puntos para la gráfica

# ------- Definir la función h(x) -------
x = sp.symbols('x')
h_sim = sp.sin(3*x) * (1 - x**2)     
h_mpm = sp.lambdify(x, h_sim, 'mpmath')
h_np = sp.lambdify(x, h_sim, 'numpy')

# ------- Construir aproximación por Legendre (proyección L2 sin peso) -------
def P_sim(k): return sp.legendre(k, x)
def P_mpm(k): return sp.lambdify(x, sp.legendre(k, x), 'mpmath')

coef_legendre = []
for k in range(grado_N + 1):
    Pk = P_mpm(k)
    integral = mp.quad(lambda t: h_mpm(t) * Pk(t), [intervalo[0], intervalo[1]])
    ck = (2*k + 1)/2 * integral
    coef_legendre.append(ck)

aprox_leg_sim = sum([sp.N(coef_legendre[k]) * P_sim(k) for k in range(grado_N + 1)])
aprox_leg_sim = sp.expand(aprox_leg_sim)
aprox_leg_np = sp.lambdify(x, aprox_leg_sim, 'numpy')

# ------- Construir aproximación por Chebyshev  -------
# usando la transformación x = cos(theta)
def T_sim(k): return sp.chebyshevt(k, x)
def f_theta(th): return h_mpm(mp.cos(th))

coef_cheb = []
for k in range(grado_N + 1):
    if k == 0:
        I = mp.quad(lambda th: f_theta(th), [0, mp.pi])
        ak = (1/mp.pi) * I
    else:
        I = mp.quad(lambda th: f_theta(th) * mp.cos(k*th), [0, mp.pi])
        ak = (2/mp.pi) * I
    coef_cheb.append(ak)

aprox_cheb_sim = sum([sp.N(coef_cheb[k]) * T_sim(k) for k in range(grado_N + 1)])
aprox_cheb_sim = sp.expand(aprox_cheb_sim)
aprox_cheb_np = sp.lambdify(x, aprox_cheb_sim, 'numpy')

# ------- Preparar datos para graficar -------
xx = np.linspace(intervalo[0], intervalo[1], puntos)
y_exact = h_np(xx)
y_leg = aprox_leg_np(xx)
y_cheb = aprox_cheb_np(xx)

# ------- Figura con dos subplots (izquierda: función, derecha: comparativa) -------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios':[1,1.4]})

# Izquierda: función original sola (línea gruesa)
ax1.plot(xx, y_exact, color='maroon', linewidth=2.2, label='h(x) original')
ax1.set_title('Función original')
ax1.set_xlabel('x')
ax1.set_ylim(np.min(y_exact)*1.05, np.max(y_exact)*1.05)
ax1.grid(True, linestyle=':', linewidth=0.6)
ax1.legend(loc='upper left')

# Derecha: función original + 2 aproximaciones
ax2.plot(xx, y_exact, color='red', linewidth=1.8, label='h(x) original')
ax2.plot(xx, y_leg, linestyle='--', linewidth=1.6, label=f'Aproximación Legendre (grado {grado_N})', alpha=0.95)
ax2.plot(xx, y_cheb, linestyle=':', linewidth=1.6, label=f'Aproximación Chebyshev T (grado {grado_N})', alpha=0.95)

ax2.set_title('Aproximaciones')
ax2.set_xlabel('x')
ax2.set_ylim(np.min(y_exact)*1.05, np.max(y_exact)*1.05)
ax2.grid(True, linestyle=':', linewidth=0.6)
ax2.legend(loc='upper left')

for ax in (ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out')

plt.tight_layout()
plt.show()


