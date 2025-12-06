import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Dict

st.set_page_config(page_title="Los Convergentes", layout="wide")

# ======================================
# DATA CLASS PARA RESULTADOS
# ======================================
@dataclass
class IterationResult:
    method: str
    xs: List[float]
    errors: List[float]
    converged: bool
    reason: str
    runtime_ms: float
    order: Optional[float] = None  # 👈 orden de convergencia estimado


# ======================================
# UTILIDAD: ORDEN DE CONVERGENCIA
# ======================================
def estimate_order(errors: List[float]) -> Optional[float]:
    """Estima el orden de convergencia usando la relación:
       e_{k+1} ≈ C * e_k^p"""
    es = [e for e in errors if e > 0]
    if len(es) < 3:
        return None

    ps = []
    for i in range(1, len(es) - 1):
        if es[i] == 0 or es[i - 1] == 0:
            continue
        try:
            num = math.log(es[i + 1] / es[i])
            den = math.log(es[i] / es[i - 1])
            if den != 0:
                ps.append(num / den)
        except ValueError:
            # Por si hay logs de valores muy pequeños / negativos numéricamente
            continue

    if len(ps) == 0:
        return None

    return sum(ps) / len(ps)


# ======================================
# MÉTODOS ITERATIVOS
# ======================================

# ---- 1. Punto fijo ----
def fixed_point(g, f, x0, tol, max_iter) -> IterationResult:
    xs = [x0]
    errors = [abs(f(x0))]
    start = time.perf_counter()
    reason = ""

    for _ in range(max_iter):
        x1 = g(x0)
        err = abs(f(x1))

        xs.append(x1)
        errors.append(err)

        if err < tol:
            reason = "|f(x_n)| < tol"
            break

        if abs(x1 - x0) < tol:
            reason = "|x_n - x_{n-1}| < tol"
            break

        x0 = x1
    else:
        reason = "Iteraciones máximas alcanzadas"

    runtime = (time.perf_counter() - start) * 1000
    order = estimate_order(errors)
    converged = reason != "Iteraciones máximas alcanzadas"
    return IterationResult("Punto Fijo", xs, errors, converged, reason, runtime, order)


# ---- 2. Bisección ----
def bisection(f, a, b, tol, max_iter) -> IterationResult:
    if f(a) * f(b) > 0:
        raise ValueError("En Bisección, f(a) y f(b) deben tener signos opuestos.")

    xs, errors = [], []
    start = time.perf_counter()
    prev = None
    reason = ""

    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)

        xs.append(m)
        errors.append(abs(fm))

        if abs(fm) < tol:
            reason = "|f(x_n)| < tol"
            break

        if prev is not None and abs(m - prev) < tol:
            reason = "|x_n - x_{n-1}| < tol"
            break

        if f(a) * fm < 0:
            b = m
        else:
            a = m

        prev = m
    else:
        reason = "Iteraciones máximas alcanzadas"

    runtime = (time.perf_counter() - start) * 1000
    order = estimate_order(errors)
    converged = reason != "Iteraciones máximas alcanzadas"
    return IterationResult("Bisección", xs, errors, converged, reason, runtime, order)


# ---- 3. Regla Falsa ----
def false_position(f, a, b, tol, max_iter) -> IterationResult:
    if f(a) * f(b) > 0:
        raise ValueError("En Regla Falsa, f(a) y f(b) deben tener signos opuestos.")

    xs, errors = [], []
    start = time.perf_counter()
    prev = None
    reason = ""

    for _ in range(max_iter):
        fa, fb = f(a), f(b)
        x = b - fb * (b - a) / (fb - fa)
        fx = f(x)

        xs.append(x)
        errors.append(abs(fx))

        if abs(fx) < tol:
            reason = "|f(x_n)| < tol"
            break

        if prev is not None and abs(x - prev) < tol:
            reason = "|x_n - x_{n-1}| < tol"
            break

        if fa * fx < 0:
            b = x
        else:
            a = x

        prev = x
    else:
        reason = "Iteraciones máximas alcanzadas"

    runtime = (time.perf_counter() - start) * 1000
    order = estimate_order(errors)
    converged = reason != "Iteraciones máximas alcanzadas"
    return IterationResult("Regla Falsa", xs, errors, converged, reason, runtime, order)


# ---- 4. Newton–Raphson ----
def newton(f, df, x0, tol, max_iter) -> IterationResult:
    xs = [x0]
    errors = [abs(f(x0))]
    start = time.perf_counter()
    reason = ""

    for _ in range(max_iter):
        fx = f(x0)
        dfx = df(x0)

        if dfx == 0:
            reason = "f'(x_n) = 0 (no se puede continuar)"
            break

        x1 = x0 - fx / dfx

        xs.append(x1)
        errors.append(abs(f(x1)))

        if abs(f(x1)) < tol:
            reason = "|f(x_n)| < tol"
            break

        if abs(x1 - x0) < tol:
            reason = "|x_n - x_{n-1}| < tol"
            break

        x0 = x1
    else:
        reason = "Iteraciones máximas alcanzadas"

    runtime = (time.perf_counter() - start) * 1000
    order = estimate_order(errors)
    converged = reason != "Iteraciones máximas alcanzadas"
    return IterationResult("Newton-Raphson", xs, errors, converged, reason, runtime, order)


# ---- 5. Secante ----
def secant(f, x0, x1, tol, max_iter) -> IterationResult:
    xs = [x0, x1]
    errors = [abs(f(x0)), abs(f(x1))]
    start = time.perf_counter()
    reason = ""

    for _ in range(max_iter):
        f0, f1 = f(x0), f(x1)
        if (f1 - f0) == 0:
            reason = "División por cero en Secante"
            break

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)

        xs.append(x2)
        errors.append(abs(f(x2)))

        if abs(f(x2)) < tol:
            reason = "|f(x_n)| < tol"
            break
        if abs(x2 - x1) < tol:
            reason = "|x_n - x_{n-1}| < tol"
            break

        x0, x1 = x1, x2
    else:
        reason = "Iteraciones máximas alcanzadas"

    runtime = (time.perf_counter() - start) * 1000
    order = estimate_order(errors)
    converged = reason != "Iteraciones máximas alcanzadas"
    return IterationResult("Secante", xs, errors, converged, reason, runtime, order)


# ======================================
# FUNCIONES DE PRUEBA
# ======================================
FUNCTIONS: Dict[str, Dict] = {
    "x^3 - x - 1": {
        "f": lambda x: x**3 - x - 1,
        "df": lambda x: 3 * x**2 - 1,
        "g": lambda x: (x + 1)**(1/3),  # punto fijo
    },
    "cos(x) - x": {
        "f": lambda x: math.cos(x) - x,
        "df": lambda x: -math.sin(x) - 1,
        "g": lambda x: math.cos(x),
    },
    "x^2 - 2": {
        "f": lambda x: x**2 - 2,
        "df": lambda x: 2 * x,
        "g": lambda x: math.sqrt(2),   # constante: mal g, pero sirve como ejemplo simple
    },
}

# ======================================
# INTERFAZ STREAMLIT
# ======================================
st.title("Los Convergentes – Comparador de Métodos Iterativos")

st.markdown("""
Aplicación para comparar **métodos iterativos de raíces**:

- Punto Fijo  
- Bisección  
- Regla Falsa  
- Newton–Raphson  
- Secante  

Se analizan: **error vs iteraciones, orden de convergencia, criterios de parada
y eficiencia computacional**.
""")

st.sidebar.header("Parámetros")

func_name = st.sidebar.selectbox("Función", list(FUNCTIONS.keys()))
f = FUNCTIONS[func_name]["f"]
df = FUNCTIONS[func_name]["df"]
g = FUNCTIONS[func_name]["g"]

tol = st.sidebar.number_input("Tolerancia", value=1e-6, format="%.1e")
max_iter = st.sidebar.number_input("Máx iteraciones", value=50)

st.sidebar.subheader("Parámetros iniciales")
a = st.sidebar.number_input("a (intervalo)", value=0.0)
b = st.sidebar.number_input("b (intervalo)", value=2.0)
x0 = st.sidebar.number_input("x0", value=1.0)
x1 = st.sidebar.number_input("x1 (secante)", value=2.0)

st.sidebar.subheader("Métodos a usar")
use_fp = st.sidebar.checkbox("Punto Fijo", True)
use_bi = st.sidebar.checkbox("Bisección", True)
use_rf = st.sidebar.checkbox("Regla Falsa", True)
use_nr = st.sidebar.checkbox("Newton-Raphson", True)
use_sc = st.sidebar.checkbox("Secante", True)

run = st.sidebar.button("Ejecutar")


# ======================================
# EJECUCIÓN Y VISUALIZACIONES
# ======================================
if run:
    results: List[IterationResult] = []

    if use_fp:
        results.append(fixed_point(g, f, x0, tol, max_iter))
    if use_bi:
        try:
            results.append(bisection(f, a, b, tol, max_iter))
        except Exception as e:
            st.error(f"Bisección: {e}")
    if use_rf:
        try:
            results.append(false_position(f, a, b, tol, max_iter))
        except Exception as e:
            st.error(f"Regla Falsa: {e}")
    if use_nr:
        results.append(newton(f, df, x0, tol, max_iter))
    if use_sc:
        results.append(secant(f, x0, x1, tol, max_iter))

    if not results:
        st.warning("No se ejecutó ningún método. Revisa los parámetros.")
    else:
        import pandas as pd

        # -------------------------
        # TABLA RESUMEN
        # -------------------------
        st.subheader("Resumen numérico de los métodos")

        df_summary = pd.DataFrame({
            "Método": [r.method for r in results],
            "Convergió": ["Sí" if r.converged else "No" for r in results],
            "Iteraciones": [len(r.xs) for r in results],
            "Aproximación final": [r.xs[-1] for r in results],
            "Error final |f(x_n)|": [r.errors[-1] for r in results],
            "Orden estimado": [r.order for r in results],
            "Tiempo (ms)": [r.runtime_ms for r in results],
            "Criterio de parada": [r.reason for r in results],
        })

        st.dataframe(
            df_summary.style.format(
                {
                    "Aproximación final": "{:.8f}",
                    "Error final |f(x_n)|": "{:.2e}",
                    "Orden estimado": "{:.3f}",
                    "Tiempo (ms)": "{:.3f}",
                }
            )
        )

        # -------------------------
        # EXPLICACIÓN DE CONVERGENCIA
        # -------------------------
        st.markdown("""
## 📈 Interpretación de la convergencia

En esta aplicación se visualiza la convergencia de cada método de tres formas:

1. **Error vs Iteración (escala logarítmica)**  
   - Muestra cómo disminuye el error |f(xₙ)|.  
   - Si la curva desciende hacia 0 → el método **converge**.  
   - Mientras más rápido descienda, mayor **tasa de convergencia**.

2. **Secuencia de aproximaciones xₙ**  
   - Muestra cómo los valores xₙ se acercan a la raíz.  
   - Si se estabilizan alrededor de un valor → el método converge.

3. **Orden de convergencia estimado p**  
   - Se calcula a partir de los errores.  
   - Valores típicos:
       - Bisección: p ≈ 1 (lineal)  
       - Regla Falsa: p ≈ 1  
       - Secante: p ≈ 1.6  
       - Newton: p ≈ 2 (cuadrática)
""")

        # -------------------------
        # GRÁFICA: ERROR VS ITERACIÓN
        # -------------------------
        st.subheader("Error vs Iteración (tasas de convergencia)")

        fig_err, ax_err = plt.subplots()
        for r in results:
            ax_err.semilogy(r.errors, marker="o", label=r.method)
        ax_err.set_xlabel("Iteración")
        ax_err.set_ylabel("Error |f(x_n)|")
        ax_err.grid(True, which="both", ls="--")
        ax_err.legend()
        st.pyplot(fig_err)

        # -------------------------
        # GRÁFICA: x_n VS ITERACIÓN
        # -------------------------
        st.subheader("Secuencia de aproximaciones xₙ")

        fig_x, ax_x = plt.subplots()
        for r in results:
            ax_x.plot(range(len(r.xs)), r.xs, marker="o", label=r.method)
        ax_x.set_xlabel("Iteración")
        ax_x.set_ylabel("x_n")
        ax_x.grid(True, ls="--")
        ax_x.legend()
        st.pyplot(fig_x)

        #grafica
        st.subheader("Ordenes de convergencia estimados")

        fig_p, ax_p = plt.subplots()
        methods = [r.method for r in results]
        orders = [r.order if r.order is not None else 0 for r in results]
        ax_p.bar(methods, orders)
        ax_p.set_ylabel("p (orden estimado)")
        ax_p.grid(True, axis="y", ls="--")
        st.pyplot(fig_p)

else:
    st.info("Configura los parámetros en el panel lateral y presiona **Ejecutar** para comparar los métodos.")
