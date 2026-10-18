"""
Tarea 1 – Parte 3: Dashboard interactivo
¿Pueden convertir titulares peruanos en un clasificador de noticias?

Ejecutar en local:   streamlit run app.py
"""
import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ── Rutas (relativas a este archivo, funcionan desde cualquier carpeta) ────
BASE = Path(__file__).parent
RUTA_NOTICIAS = BASE / "datos" / "noticias.csv"
RUTA_LLM = BASE / "datos" / "etiquetas_llm.csv"
RUTA_METRICAS = BASE / "modelo" / "metricas.json"
RUTA_MODELO = BASE / "modelo"

# Si los pesos no están en la carpeta modelo/ (no se suben a GitHub por pesar >100 MB),
# el modelo se descarga desde Hugging Face Hub. Cambia este valor por tu repositorio.
MODELO_HF = os.environ.get("MODELO_HF", "TU_USUARIO_HF/clasificador-noticias-elcomercio")

CATEGORIAS = ["politica", "economia", "deportes", "ciencia_tecnologia"]
NOMBRES = {"politica": "🏛️ Política", "economia": "💰 Economía",
           "deportes": "⚽ Deportes", "ciencia_tecnologia": "🔬 Ciencia y tecnología"}
COLORES = {"politica": "#636EFA", "economia": "#EF553B",
           "deportes": "#00CC96", "ciencia_tecnologia": "#AB63FA"}

st.set_page_config(page_title="Clasificador de noticias peruanas", page_icon="📰", layout="wide")


# ── Carga de datos y modelo (en caché para que la app sea rápida) ──────────
@st.cache_data
def cargar_datos():
    df = pd.read_csv(RUTA_NOTICIAS).fillna("")
    if RUTA_LLM.exists():
        llm = pd.read_csv(RUTA_LLM)
        if len(llm) == len(df):
            df["etiqueta_llm"] = llm["etiqueta_llm"].values
    return df


@st.cache_data
def cargar_metricas():
    if RUTA_METRICAS.exists():
        return json.loads(RUTA_METRICAS.read_text(encoding="utf-8"))
    return {}


def fuente_modelo():
    """Usa la carpeta local si tiene los pesos; si no, el repositorio de Hugging Face Hub."""
    tiene_pesos = any(RUTA_MODELO.glob("*.safetensors")) or any(RUTA_MODELO.glob("*.bin"))
    return str(RUTA_MODELO) if tiene_pesos else MODELO_HF


@st.cache_resource(show_spinner="Cargando el modelo (solo la primera vez)…")
def cargar_clasificador():
    from transformers import pipeline
    return pipeline("text-classification", model=fuente_modelo(), top_k=None)


def predecir(texto):
    """Devuelve un DataFrame con la probabilidad de cada categoría, de mayor a menor."""
    salida = cargar_clasificador()(texto)
    if salida and isinstance(salida[0], list):          # algunas versiones anidan la lista
        salida = salida[0]
    probs = pd.DataFrame(salida).rename(columns={"label": "categoria", "score": "probabilidad"})
    return probs.sort_values("probabilidad", ascending=False).reset_index(drop=True)


df = cargar_datos()
metricas = cargar_metricas()

# ── 1. Encabezado y contexto ───────────────────────────────────────────────
st.title("📰 ¿Pueden convertir titulares peruanos en un clasificador de noticias?")
st.markdown(
    f"""
**Resumen metodológico:** se recolectaron **{len(df)} noticias de El Comercio** mediante sus RSS oficiales
y sitemaps, y se normalizaron en 4 categorías a partir de la sección del diario. Un LLM
(**{metricas.get('modelo_llm', 'Gemini')}**) etiquetó todo el dataset (supervisión débil) y se validó con una
auditoría humana de 20 casos. Con esas etiquetas se hizo fine-tuning de
**{metricas.get('modelo_base', 'DistilBERT multilingüe')}**, evaluado en un conjunto de test reservado (15 %).
"""
)
st.divider()

# ── 2. Filtros en la barra lateral ─────────────────────────────────────────
st.sidebar.header("🔎 Filtros")
medios = sorted(df["medio"].unique())
sel_medios = st.sidebar.multiselect("Medio", medios, default=medios)
sel_cats = st.sidebar.multiselect("Categoría", CATEGORIAS, default=CATEGORIAS,
                                  format_func=lambda c: NOMBRES[c])
filtrado = df[df["medio"].isin(sel_medios) & df["etiqueta"].isin(sel_cats)]
st.sidebar.caption(f"Mostrando **{len(filtrado)}** de {len(df)} noticias")

# ── 3. Indicadores clave ───────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Noticias recolectadas", f"{len(df)}",
          delta=f"{len(filtrado)} con los filtros actuales", delta_color="off")
f1 = metricas.get("f1_macro_test")
f1_zs = metricas.get("f1_macro_zero_shot_40")
c2.metric("F1 macro del modelo en test", f"{f1:.2f}" if f1 is not None else "—",
          delta=f"{f1 - f1_zs:+.2f} vs. zero-shot" if f1 is not None and f1_zs is not None else None)
acuerdo = metricas.get("acuerdo_llm_humano")
c3.metric("Concordancia LLM vs. auditoría humana", f"{acuerdo:.0%}" if acuerdo is not None else "—",
          delta="20 casos revisados a mano", delta_color="off")

if not metricas:
    st.warning("No se encontró `modelo/metricas.json`. Ejecuta primero `02_clasificacion.ipynb`.")

st.divider()

# ── 4. Visualización y exploración de datos ────────────────────────────────
col_graf, col_tabla = st.columns([2, 3])

with col_graf:
    conteo = filtrado["etiqueta"].value_counts().reindex(sel_cats, fill_value=0).reset_index()
    conteo.columns = ["categoria", "n"]
    if conteo["n"].sum() > 0:
        top = conteo.loc[conteo["n"].idxmax()]
        titulo = f"{NOMBRES[top.categoria]} concentra el {top.n / conteo.n.sum():.0%} de las noticias filtradas"
    else:
        titulo = "Sin noticias para los filtros elegidos"
    fig = px.bar(conteo, x="categoria", y="n", color="categoria", text="n", title=titulo,
                 color_discrete_map=COLORES, labels={"categoria": "Categoría", "n": "N.º de noticias"})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

with col_tabla:
    st.subheader("Noticias")
    columnas = [c for c in ["titular", "etiqueta", "etiqueta_llm", "seccion_original", "fecha", "url"]
                if c in filtrado.columns]
    st.dataframe(filtrado[columnas], width="stretch", height=360, hide_index=True,
                 column_config={"url": st.column_config.LinkColumn("url", display_text="Abrir")})
    st.download_button("⬇️ Descargar CSV filtrado",
                       filtrado.to_csv(index=False).encode("utf-8"),
                       file_name="noticias_filtradas.csv", mime="text/csv")

st.divider()

# ── 5. Probador interactivo en vivo ────────────────────────────────────────
st.header("🧪 Prueba el clasificador en vivo")
titular = st.text_area("Escribe o pega un titular de noticia:",
                       placeholder="Ej.: Congreso aprueba nuevo retiro de fondos de AFP", height=90)

if st.button("Clasificar", type="primary"):
    if not titular.strip():
        st.warning("Escribe un titular antes de clasificar.")
    else:
        try:
            probs = predecir(titular.strip())
        except Exception as e:
            st.error(f"No se pudo cargar el modelo desde `{fuente_modelo()}`. "
                     f"Revisa la sección *Modelo* del README.\n\nDetalle: {e}")
        else:
            mejor = probs.iloc[0]
            st.success(f"**Categoría predicha:** {NOMBRES.get(mejor.categoria, mejor.categoria)} "
                       f"— probabilidad **{mejor.probabilidad:.1%}**")
            if mejor.probabilidad < 0.5:
                st.info("Confianza baja: el titular podría pertenecer a más de una categoría.")
            fig_p = px.bar(probs, x="probabilidad", y="categoria", orientation="h",
                           color="categoria", color_discrete_map=COLORES, text_auto=".1%",
                           labels={"probabilidad": "Probabilidad", "categoria": ""})
            fig_p.update_layout(showlegend=False, xaxis_range=[0, 1], height=260,
                                yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_p, width="stretch")

st.caption("Tarea 1 · Datos: El Comercio (elcomercio.pe) · Modelo: Transformer fine-tuned con etiquetas de un LLM")
