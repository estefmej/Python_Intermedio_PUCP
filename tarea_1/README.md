# 📰 Tarea 1 – Noticias peruanas: del titular al clasificador

## ❓ Pregunta del proyecto
**¿Pueden convertir titulares peruanos en un clasificador de noticias?**

El proyecto recolecta noticias reales de **El Comercio**, las etiqueta de forma semi-supervisada con un LLM, entrena un Transformer ligero y lo despliega en un dashboard de Streamlit, donde cualquier persona puede escribir un titular y ver la categoría predicha junto con su probabilidad.

---

## 🗞️ Medio y fechas de extracción
| Campo | Valor |
|---|---|
| Opción | **B – El Comercio** (`elcomercio.pe`) |
| Fecha de extracción | 2026-10-17 |
| Fuentes | RSS oficiales por sección (Google News feed y Arc) + sitemaps web/news declarados en `robots.txt` |
| Noticias finales | **388** en 4 categorías: `politica`, `economia`, `deportes`, `ciencia_tecnologia` |
| Buenas prácticas | `User-Agent` descriptivo, pausa de 1.5 s entre solicitudes, máximo 1 reintento, respeto de `robots.txt` |

---

## ⚙️ Metodología
1. **Recolección y limpieza** (`01_recoleccion.ipynb`): extracción vía RSS y sitemaps, eliminación de duplicados y vacíos, mapeo `seccion_original → etiqueta` y 3 gráficos de EDA.
2. **Clasificación** (`02_clasificacion.ipynb`):
   - Línea base **zero-shot** con `Recognai/bert-base-spanish-wwm-cased-xnli`.
   - **Supervisión débil** con Gemini (`gemini-flash-latest`) y **auditoría humana** de 20 casos.
   - Split **70/15/15** estratificado con `random_state=42`.
   - **Fine-tuning** de `distilbert-base-multilingual-cased` (3 épocas, batch 16, métrica F1 macro).
   - Evaluación final en el **test reservado**.
3. **Dashboard** (`app.py`): filtros, métricas, gráfico, tabla descargable y probador en vivo.

### Resultados (conjunto de test)
| Modelo | Accuracy | F1 macro | Latencia (ms/titular) |
|---|---|---|---|
| Zero-shot (bert-base-spanish-xnli) | 0.593 | 0.558 | 166.3 |
| LLM débil (Gemini) | 0.966 | 0.966 | 1 019.7 |
| **Transformer fine-tuned (DistilBERT)** | **0.847** | **0.850** | **20.0** |

**Modelo elegido para producción:** el Transformer fine-tuned. El LLM es más preciso, pero el fine-tuned es ~50 veces más rápido, corre localmente y no depende de una API con cuota.

Auditoría humana del LLM: **20/20 = 100 %** de acuerdo.

---

## 🔍 Hallazgos analíticos
1. **El zero-shot tiende a clasificar todo como política.** Sin entrenamiento, el modelo NLI alcanzó un F1 macro de solo 0.56 en 40 titulares: acertó el 100 % de política, pero apenas el 30 % de economía y de ciencia y tecnología. Cualquier mención al Gobierno, leyes o instituciones lo empuja hacia `politica`. El fine-tuning con etiquetas del LLM corrige este sesgo y eleva el F1 macro a 0.85 en el test reservado.
2. **Las noticias de medio ambiente se confunden con economía.** De los 9 errores del modelo en test, 4 son noticias de ciencia sobre medio ambiente predichas como economía (por ejemplo, *"la fauna silvestre cae 73%"*). Usan cifras, porcentajes y verbos como "cae" o "disminuyó", típicos de los titulares económicos. A la inversa, *"Premio Nobel de Economía"* o *"IA y Sandbox regulatorio"* se predijeron como ciencia por palabras como "Nobel" o "IA". Por eso `economia` tiene el F1 más bajo (0.79) y `deportes` el más alto (0.90), gracias a su vocabulario muy específico.

## ⚠️ Limitaciones técnicas
1. **Dataset pequeño y de un periodo corto.** Las 388 noticias provienen de un solo medio y de pocos días, así que el modelo aprende el vocabulario de la coyuntura de esas fechas (nombres de políticos, torneos en curso). Su desempeño puede bajar con noticias de otros periodos o de otros diarios.
2. **Dependencia de servicios externos y recursos de cómputo.** El etiquetado depende de la cuota gratuita de la API de Gemini, con límites por minuto (errores 429/503) y modelos que se retiran: `gemini-2.5-flash` dejó de estar disponible (error 404) y hubo que cambiar a `gemini-flash-latest`. Además, los pesos del modelo pesan unos 540 MB, por lo que no se pueden subir a GitHub y deben alojarse en Hugging Face Hub.

---

## ▶️ Cómo ejecutar en local
Requisitos: **Python 3.14**.

```bash
# 1. Clonar el repositorio y entrar a la carpeta
git clone https://github.com/estefmej/Python_Intermedio_PUCP.git
cd Python_Intermedio_PUCP/tarea_1

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar el dashboard
streamlit run app.py
```

La app se abre en `http://localhost:8501`.

### 🤖 Modelo
Los pesos del modelo (`model.safetensors`, ~540 MB) **no están en GitHub** porque superan el límite de 100 MB. La app los busca así:
1. Si existen en `modelo/` (después de ejecutar `02_clasificacion.ipynb`), usa esa carpeta.
2. Si no, los descarga desde Hugging Face Hub: **`TU_USUARIO_HF/clasificador-noticias-elcomercio`**.

### 🔑 Re-ejecutar los notebooks (opcional)
Las etiquetas del LLM ya están guardadas en `datos/etiquetas_llm.csv`, así que **no se necesita clave de API** para ejecutar `02_clasificacion.ipynb`. Solo si se borra ese archivo, el notebook pedirá una `GEMINI_API_KEY` (gratuita en [aistudio.google.com](https://aistudio.google.com)), que se lee de un archivo `.env` excluido en `.gitignore`.

---

## 📁 Estructura
```
tarea_1/
├── 01_recoleccion.ipynb      # Scraping, limpieza y EDA
├── 02_clasificacion.ipynb    # Zero-shot, LLM, fine-tuning y evaluación
├── app.py                    # Dashboard de Streamlit
├── requirements.txt          # Dependencias exactas
├── README.md                 # Este reporte
├── .gitignore                # Excluye pesos del modelo y .env
├── datos/
│   ├── noticias.csv          # 388 noticias (7 columnas)
│   ├── etiquetas_llm.csv     # Etiquetas de Gemini
│   ├── auditoria_llm.csv     # Auditoría humana de 20 casos
│   ├── test.csv              # Conjunto de test reservado
│   └── raw/                  # Respaldos crudos (RSS y sitemaps)
└── modelo/
    ├── config.json           # Configuración del modelo
    ├── tokenizer*.json       # Tokenizador
    └── metricas.json         # Métricas que usa el dashboard
```
