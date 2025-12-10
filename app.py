import streamlit as st
import math
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass
from typing import List, Optional, Dict

st.set_page_config(page_title="Los Convergentes", layout="wide")



@dataclass
class ResultadoIteracion:
    metodo: str
    valores_x: List[float]
    errores: List[float]
    convergio: bool
    criterio: str
    tiempo_ms: float
    orden: Optional[float] = None


def estimar_orden(errores: List[float]) -> Optional[float]:
    errores_positivos = [e for e in errores if e > 0]
    if len(errores_positivos) < 3:
        return None

    lista_p = []
    for i in range(1, len(errores_positivos) - 1):
        if errores_positivos[i] == 0 or errores_positivos[i - 1] == 0:
            continue
        try:
            num = math.log(errores_positivos[i + 1] / errores_positivos[i])
            den = math.log(errores_positivos[i] / errores_positivos[i - 1])
            if den != 0:
                lista_p.append(num / den)
        except ValueError:
            continue

    if len(lista_p) == 0:
        return None

    return sum(lista_p) / len(lista_p)

# 1. METODO DE PUNTO FIJO
def metodo_punto_fijo(g, f, x0, tolerancia, max_iteraciones) -> ResultadoIteracion:
    valores_x = [x0]
    errores = [abs(f(x0))]
    inicio = time.perf_counter()
    criterio = ""

    for _ in range(max_iteraciones):
        x1 = g(x0)
        error_actual = abs(f(x1))

        valores_x.append(x1)
        errores.append(error_actual)

        if error_actual < tolerancia:
            criterio = "|f(x_n)| < tolerancia"
            break

        if abs(x1 - x0) < tolerancia:
            criterio = "|x_n - x_{n-1}| < tolerancia"
            break

        x0 = x1
    else:
        criterio = "Iteraciones maximas alcanzadas"

    tiempo_ms = (time.perf_counter() - inicio) * 1000
    orden = estimar_orden(errores)
    convergio = criterio != "Iteraciones maximas alcanzadas"
    return ResultadoIteracion("Punto Fijo", valores_x, errores, convergio, criterio, tiempo_ms, orden)



# 2. METODO DE BISECCION

def metodo_biseccion(f, a, b, tolerancia, max_iteraciones) -> ResultadoIteracion:
    if f(a) * f(b) > 0:
        raise ValueError("En Bisección, f(a) y f(b) deben tener signos opuestos.")

    valores_x, errores = [], []
    inicio = time.perf_counter()
    x_anterior = None
    criterio = ""

    for _ in range(max_iteraciones):
        m = 0.5 * (a + b)
        fm = f(m)

        valores_x.append(m)
        errores.append(abs(fm))

        if abs(fm) < tolerancia:
            criterio = "|f(x_n)| < tolerancia"
            break

        if x_anterior is not None and abs(m - x_anterior) < tolerancia:
            criterio = "|x_n - x_{n-1}| < tolerancia"
            break

        if f(a) * fm < 0:
            b = m
        else:
            a = m

        x_anterior = m
    else:
        criterio = "Iteraciones maximas alcanzadas"

    tiempo_ms = (time.perf_counter() - inicio) * 1000
    orden = estimar_orden(errores)
    convergio = criterio != "Iteraciones maximas alcanzadas"
    return ResultadoIteracion("Biseccion", valores_x, errores, convergio, criterio, tiempo_ms, orden)



# 3. METODO DE REGLA FALSA o FALSA POSICION

def metodo_regla_falsa(f, a, b, tolerancia, max_iteraciones) -> ResultadoIteracion:
    if f(a) * f(b) > 0:
        raise ValueError("En Regla Falsa, f(a) y f(b) deben tener signos opuestos.")

    valores_x, errores = [], []
    inicio = time.perf_counter()
    x_anterior = None
    criterio = ""

    for _ in range(max_iteraciones):
        fa, fb = f(a), f(b)
        x = b - fb * (b - a) / (fb - fa)
        fx = f(x)

        valores_x.append(x)
        errores.append(abs(fx))

        if abs(fx) < tolerancia:
            criterio = "|f(x_n)| < tolerancia"
            break

        if x_anterior is not None and abs(x - x_anterior) < tolerancia:
            criterio = "|x_n - x_{n-1}| < tolerancia"
            break

        if fa * fx < 0:
            b = x
        else:
            a = x

        x_anterior = x
    else:
        criterio = "Iteraciones maximas alcanzadas"

    tiempo_ms = (time.perf_counter() - inicio) * 1000
    orden = estimar_orden(errores)
    convergio = criterio != "Iteraciones maximas alcanzadas"
    return ResultadoIteracion("Regla Falsa", valores_x, errores, convergio, criterio, tiempo_ms, orden)


# 4. METODO DE NEWTON–RAPHSON

def metodo_newton_raphson(f, df, x0, tolerancia, max_iteraciones) -> ResultadoIteracion:
    valores_x = [x0]
    errores = [abs(f(x0))]
    inicio = time.perf_counter()
    criterio = ""

    for _ in range(max_iteraciones):
        fx = f(x0)
        dfx = df(x0)

        if dfx == 0:
            criterio = "f'(x_n) = 0 (no se puede continuar)"
            break

        x1 = x0 - fx / dfx

        valores_x.append(x1)
        errores.append(abs(f(x1)))

        if abs(f(x1)) < tolerancia:
            criterio = "|f(x_n)| < tolerancia"
            break

        if abs(x1 - x0) < tolerancia:
            criterio = "|x_n - x_{n-1}| < tolerancia"
            break

        x0 = x1
    else:
        criterio = "Iteraciones maximas alcanzadas"

    tiempo_ms = (time.perf_counter() - inicio) * 1000
    orden = estimar_orden(errores)
    convergio = criterio != "Iteraciones maximas alcanzadas"
    return ResultadoIteracion("Newton-Raphson", valores_x, errores, convergio, criterio, tiempo_ms, orden)

# 5. METODO DE SECANTE
def metodo_secante(f, x0, x1, tolerancia, max_iteraciones) -> ResultadoIteracion:
    valores_x = [x0, x1]
    errores = [abs(f(x0)), abs(f(x1))]
    inicio = time.perf_counter()
    criterio = ""

    for _ in range(max_iteraciones):
        f0, f1 = f(x0), f(x1)
        if (f1 - f0) == 0:
            criterio = "Division por cero en Secante"
            break

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)

        valores_x.append(x2)
        errores.append(abs(f(x2)))

        if abs(f(x2)) < tolerancia:
            criterio = "|f(x_n)| < tolerancia"
            break
        if abs(x2 - x1) < tolerancia:
            criterio = "|x_n - x_{n-1}| < tolerancia"
            break

        x0, x1 = x1, x2
    else:
        criterio = "Iteraciones maximas alcanzadas"

    tiempo_ms = (time.perf_counter() - inicio) * 1000
    orden = estimar_orden(errores)
    convergio = criterio != "Iteraciones maximas alcanzadas"
    return ResultadoIteracion("Secante", valores_x, errores, convergio, criterio, tiempo_ms, orden)



FUNCIONES: Dict[str, Dict] = {
    "x^3 - x - 1": {
        "f": lambda x: x**3 - x - 1,
        "df": lambda x: 3 * x**2 - 1,
        "g": lambda x: (x + 1)**(1/3),
    },
    "cos(x) - x": {
        "f": lambda x: math.cos(x) - x,
        "df": lambda x: -math.sin(x) - 1,
        "g": lambda x: math.cos(x),
    },
    "x^2 - 2": {
        "f": lambda x: x**2 - 2,
        "df": lambda x: 2 * x,
        "g": lambda x: math.sqrt(2),
    },
}

st.title("Los Convergentes – Comparador de Métodos Iterativos")

st.markdown("""
Aplicacion para comparar **metodos iterativos de raices**:

- Punto Fijo  
- Biseccion  
- Regla Falsa  
- Newton–Raphson  
- Secante  

Se analizan: **error vs iteraciones, orden de convergencia, criterios de parada
y eficiencia computacional**.
""")

st.sidebar.header("Parametros")

nombre_funcion = st.sidebar.selectbox("Funcion", list(FUNCIONES.keys()))
f = FUNCIONES[nombre_funcion]["f"]
df = FUNCIONES[nombre_funcion]["df"]
g = FUNCIONES[nombre_funcion]["g"]

tolerancia = st.sidebar.number_input("Tolerancia", value=1e-6, format="%.1e")
max_iteraciones = st.sidebar.number_input("Max. iteraciones", value=50)

st.sidebar.subheader("Parametros iniciales")
a = st.sidebar.number_input("a (intervalo)", value=0.0)
b = st.sidebar.number_input("b (intervalo)", value=2.0)
x0 = st.sidebar.number_input("x0", value=1.0)
x1 = st.sidebar.number_input("x1 (secante)", value=2.0)

st.sidebar.subheader("Metodos a usar")
usar_punto_fijo = st.sidebar.checkbox("Punto Fijo", True)
usar_biseccion = st.sidebar.checkbox("Biseccion", True)
usar_regla_falsa = st.sidebar.checkbox("Regla Falsa", True)
usar_newton = st.sidebar.checkbox("Newton-Raphson", True)
usar_secante = st.sidebar.checkbox("Secante", True)

boton_ejecutar = st.sidebar.button("Ejecutar")


if boton_ejecutar:
    resultados: List[ResultadoIteracion] = []

    if usar_punto_fijo:
        resultados.append(metodo_punto_fijo(g, f, x0, tolerancia, max_iteraciones))
    if usar_biseccion:
        try:
            resultados.append(metodo_biseccion(f, a, b, tolerancia, max_iteraciones))
        except Exception as e:
            st.error(f"Biseccion: {e}")
    if usar_regla_falsa:
        try:
            resultados.append(metodo_regla_falsa(f, a, b, tolerancia, max_iteraciones))
        except Exception as e:
            st.error(f"Regla Falsa: {e}")
    if usar_newton:
        resultados.append(metodo_newton_raphson(f, df, x0, tolerancia, max_iteraciones))
    if usar_secante:
        resultados.append(metodo_secante(f, x0, x1, tolerancia, max_iteraciones))

    if not resultados:
        st.warning("No se ejecutó ningun metodo. Revisa los parametros.")
    else:
        import pandas as pd

        st.subheader("Resumen numeric de los metodos")

        df_resumen = pd.DataFrame({
            "Metodo": [r.metodo for r in resultados],
            "Convergio": ["Sí" if r.convergio else "No" for r in resultados],
            "Iteraciones": [len(r.valores_x) for r in resultados],
            "Aproximacion final": [r.valores_x[-1] for r in resultados],
            "Error final |f(x_n)|": [r.errores[-1] for r in resultados],
            "Orden estimado": [r.orden for r in resultados],
            "Tiempo (ms)": [r.tiempo_ms for r in resultados],
            "Criterio de parada": [r.criterio for r in resultados],
        })

        st.dataframe(
            df_resumen.style.format(
                {
                    "Aproximacion final": "{:.8f}",
                    "Error final |f(x_n)|": "{:.2e}",
                    "Orden estimado": "{:.3f}",
                    "Tiempo (ms)": "{:.3f}",
                }
            )
        )

        st.markdown("""
## Interpretacion de la convergencia

En esta aplicacion se visualiza la convergencia de cada metodo de tres formas:

1. **Error vs Iteracion (escala logaritmica)**  
   - Muestra como disminuye el error |f(x_n)|.  
   - Si la curva desciende hacia 0 → el metodo converge.  
   - Mientras mas rapido descienda, mayor tasa de convergencia.

2. **Secuencia de aproximaciones x_n**  
   - Muestra como los valores x_n se acercan a la raiz.  
   - Si se estabilizan alrededor de un valor → el metodo converge.

3. **Orden de convergencia estimado p**  
   - Se calcula a partir de los errores.  
   - Valores típicos:
       - Bisección: p ≈ 1 (lineal)  
       - Regla Falsa: p ≈ 1  
       - Secante: p ≈ 1.6  
       - Newton: p ≈ 2 (cuadrática)
""")

        #graficos
        st.subheader("Error vs Iteracion (tasas de convergencia)")

        fig_err, ax_err = plt.subplots()
        for r in resultados:
            ax_err.semilogy(r.errores, marker="o", label=r.metodo)
        ax_err.set_xlabel("Iteracion")
        ax_err.set_ylabel("Error |f(x_n)|")
        ax_err.grid(True, which="both", ls="--")
        ax_err.legend()
        st.pyplot(fig_err)

        
        st.subheader("Secuencia de aproximaciones x_n")

        fig_x, ax_x = plt.subplots()
        for r in resultados:
            ax_x.plot(range(len(r.valores_x)), r.valores_x, marker="o", label=r.metodo)
        ax_x.set_xlabel("Iteracion")
        ax_x.set_ylabel("x_n")
        ax_x.grid(True, ls="--")
        ax_x.legend()
        st.pyplot(fig_x)

        
        st.subheader("Ordenes de convergencia estimados")

        fig_p, ax_p = plt.subplots()
        nombres_metodos = [r.metodo for r in resultados]
        ordenes = [r.orden if r.orden is not None else 0 for r in resultados]
        ax_p.bar(nombres_metodos, ordenes)
        ax_p.set_ylabel("p (orden estimado)")
        ax_p.grid(True, axis="y", ls="--")
        st.pyplot(fig_p)

else:
    st.info("Configura los parametros en el panel lateral y presiona **Ejecutar** para comparar los metodos.")
