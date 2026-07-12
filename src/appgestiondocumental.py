import streamlit as st
import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ==============================
# CONFIGURACIÓN
# ==============================


st.set_page_config(page_title="Gestión de Documental IA", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f4f7f9; }
    h1 { color: #1E3A8A; font-family: 'Segoe UI', sans-serif; }
    .metric-card {
        background-color: white; padding: 15px; border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #1E3A8A;
    }
    </style>
    """, unsafe_allow_html=True)

## st.title("🛡️ Sistema de Gestión de Reclamos con IA")

## st.title("🤖 BTG - Chatbot Gestión de Reclamos IA")

# ==============================
# API KEY
# ==============================

OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

    # ==============================
    # CARGAR DATASET
    # ==============================

@st.cache_data
@st.cache_data
def cargar_datos():
    df = pd.read_excel("DatasetDocumentos.xlsx", sheet_name="Dataset_Documentos")
    df.columns = df.columns.str.strip()

    columnas_criticas = [
        "documento_id",
        "tipo_documento",
        "titulo_documento",
        "fecha_documento",
        "entidad_origen",
        "texto_documento",
        "resumen_documento",
        "palabras_clave",
        "categoria",
        "estado",
        "respuesta_esperada",
        "fuente_archivo"
    ]

    for col in columnas_criticas:
        if col not in df.columns:
            df[col] = "No disponible"

    df["texto_documento"] = df["texto_documento"].fillna("").astype(str)
    return df

##########  Cargando Datos

df = cargar_datos()



# ==============================
# CARGAR MODELO DE EMBEDDINGS
# ==============================

@st.cache_resource
def cargar_modelo():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = cargar_modelo()

# ==============================
# CREAR ÍNDICE FAISS
# ==============================

@st.cache_resource
def crear_indice_faiss(textos):
    embeddings = model.encode(textos, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings


textos_documentos = df["texto_documento"].tolist()
faiss_index, embeddings = crear_indice_faiss(textos_documentos)

# ==============================
# PANEL DE MÉTRICAS
# ==============================

st.divider()

total_documentos = len(df)
documentos_vigentes = len(df[df["estado"] == "Vigente"]) if "estado" in df.columns else 0
documentos_observados = len(df[df["estado"] == "Observado"]) if "estado" in df.columns else 0

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total de documentos 📄", total_documentos)

with col2:
    st.metric("Documentos vigentes ✅", documentos_vigentes)

with col3:
    st.metric("Documentos observados ⚠️", documentos_observados)

st.divider()

# ==============================
# BANDEJA DE CONSULTA DOCUMENTAL
# ==============================

with st.expander("🔍 Bandeja de Consulta Documental", expanded=True):

    col_busq, col_tipo, col_estado = st.columns([2, 1, 1])

    with col_busq:
        busqueda = st.text_input(
            "Consulta por tema, entidad, título, palabra clave o contenido:"
        )

    with col_tipo:
        tipos = ["Todos"] + sorted(df["tipo_documento"].dropna().unique().tolist())
        filtro_tipo = st.selectbox("Tipo de documento:", tipos)

    with col_estado:
        estados = ["Todos"] + sorted(df["estado"].dropna().unique().tolist())
        filtro_estado = st.selectbox("Estado:", estados)

    ejecutar_ia = st.button("🤖 Consultar con IA")

    df_visible = df.copy()

    if filtro_tipo != "Todos":
        df_visible = df_visible[df_visible["tipo_documento"] == filtro_tipo]

    if filtro_estado != "Todos":
        df_visible = df_visible[df_visible["estado"] == filtro_estado]

    if busqueda:
        df_visible = df_visible[
            df_visible["texto_documento"].str.contains(busqueda, case=False, na=False) |
            df_visible["titulo_documento"].str.contains(busqueda, case=False, na=False) |
            df_visible["palabras_clave"].str.contains(busqueda, case=False, na=False) |
            df_visible["entidad_origen"].str.contains(busqueda, case=False, na=False)
        ]

    columnas_mostrar = [
        "documento_id",
        "tipo_documento",
        "titulo_documento",
        "entidad_origen",
        "categoria",
        "estado",
        "fuente_archivo"
    ]

    columnas_existentes = [col for col in columnas_mostrar if col in df_visible.columns]

    st.dataframe(
        df_visible[columnas_existentes].head(20),
        width="stretch"
    )

# ==============================
# CONSULTA SEMÁNTICA CON IA
# ==============================

if ejecutar_ia:

    if not busqueda:
        st.warning("Ingresa una consulta para que el agente pueda buscar información.")
        st.stop()

    if client is None:
        st.error("No se puede ejecutar IA porque no está configurada la API Key.")
        st.stop()

    with st.spinner("🤖 Buscando documentos relacionados y generando respuesta..."):

        query_embedding = model.encode([busqueda]).astype("float32")

        cantidad_resultados = 5
        distances, indices = faiss_index.search(query_embedding, cantidad_resultados)

        contexto_historico = ""

        documentos_encontrados = []

        for idx in indices[0]:
            row = df.iloc[idx]

            doc_id = row.get("documento_id", "ID desconocido")
            tipo = row.get("tipo_documento", "Sin tipo")
            titulo = row.get("titulo_documento", "Sin título")
            entidad = row.get("entidad_origen", "Sin entidad")
            categoria = row.get("categoria", "Sin categoría")
            estado = row.get("estado", "Sin estado")
            resumen = row.get("resumen_documento", "Sin resumen")
            texto = row.get("texto_documento", "")
            respuesta = row.get("respuesta_esperada", "")

            documentos_encontrados.append({
                "documento_id": doc_id,
                "tipo_documento": tipo,
                "titulo_documento": titulo,
                "entidad_origen": entidad,
                "categoria": categoria,
                "estado": estado
            })

            contexto_historico += f"""
DOCUMENTO:
- ID: {doc_id}
- Tipo: {tipo}
- Título: {titulo}
- Entidad origen: {entidad}
- Categoría: {categoria}
- Estado: {estado}
- Resumen: {resumen}
- Texto relevante: {texto[:1500]}
- Respuesta esperada: {respuesta}
"""

        prompt = f"""
Eres un asistente experto en consulta documental empresarial para BTG.

Tu función es responder consultas usando únicamente la información encontrada
en los documentos proporcionados.

CONSULTA DEL USUARIO:
{busqueda}

DOCUMENTOS RELACIONADOS:
{contexto_historico}

Responde con el siguiente formato:

1. RESPUESTA DIRECTA:
Responde claramente la consulta del usuario.

2. DOCUMENTOS RELACIONADOS:
Menciona los documentos que sustentan la respuesta.

3. SUSTENTO ENCONTRADO:
Resume qué información se encontró en los documentos.

4. ACCIÓN RECOMENDADA:
Indica qué debería hacer el usuario o el operador.

5. NIVEL DE CONFIANZA:
Indica Bajo, Medio o Alto según la información disponible.

Si no encuentras sustento suficiente, dilo claramente.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Eres un asistente experto en gestión documental empresarial."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        st.markdown("### 🧠 Respuesta del Agente Documental")
        st.success(response.choices[0].message.content)

        st.markdown("### 📌 Documentos más relacionados")

        df_resultados = pd.DataFrame(documentos_encontrados)
        st.dataframe(df_resultados, width="stretch")

# ==============================
# VISTA GENERAL DEL DATASET
# ==============================

with st.expander("📚 Ver dataset completo"):
    st.dataframe(df, width="stretch")