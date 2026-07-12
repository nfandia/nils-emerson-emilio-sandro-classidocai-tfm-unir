import streamlit as st
import pandas as pd
import re
import tempfile
from pathlib import Path
from datetime import datetime

import pdfplumber
import fitz
import pytesseract
from PIL import Image
import docx

from sentence_transformers import SentenceTransformer, util


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

ARCHIVO_TIPOS = "DatasetTipoDocumento.xlsx"
ARCHIVO_DOCUMENTOS = "DatasetDocumentos.xlsx"
HOJA_DOCUMENTOS = "Dataset_Documentos"


st.set_page_config(
    page_title="CyDoc AI Classifier",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# ESTILOS
# =========================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #020617 0%, #0f172a 45%, #1e3a8a 100%);
}

.main-header {
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 30px;
    border-radius: 24px;
    margin-bottom: 30px;
    text-align: center;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.35);
}

.main-header h1 {
    color: #38bdf8 !important;
    font-size: 48px;
    margin-bottom: 10px;
    font-weight: bold;
}

.main-header p {
    color: #e2e8f0 !important;
    font-size: 20px;
}

.glass-card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 22px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}

.glass-card h2, .glass-card h3 {
    color: #38bdf8 !important;
}

.glass-card p {
    color: white !important;
    font-size: 18px;
}

.metric-card {
    background: rgba(15,23,42,0.75);
    border-left: 5px solid #38bdf8;
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 15px;
    color: white !important;
    font-size: 17px;
}

.metric-card b {
    color: #38bdf8 !important;
    font-size: 16px;
}

div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 18px;
    padding: 20px;
}

label {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="main-header">
<h1>🤖 CyDoc AI Classifier</h1>
<p>Tipificación documental con catálogo, memoria histórica e IA semántica</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# MODELO IA
# =========================================================

@st.cache_resource
def cargar_modelo_ia():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


# =========================================================
# CARGA DE DATASETS
# =========================================================

@st.cache_data
def cargar_dataset_tipos():
    if not Path(ARCHIVO_TIPOS).exists():
        st.error(f"No se encontró el archivo {ARCHIVO_TIPOS}. Debe estar en la misma carpeta del app.py")
        st.stop()

    df = pd.read_excel(ARCHIVO_TIPOS)

    columnas_requeridas = ["tipo_documental", "palabras_clave"]

    for col in columnas_requeridas:
        if col not in df.columns:
            st.error(f"El dataset de tipos debe tener la columna: {col}")
            st.stop()

    if "texto_ejemplo" not in df.columns:
        df["texto_ejemplo"] = ""

    df["tipo_documental"] = df["tipo_documental"].astype(str).str.strip()
    df["palabras_clave"] = df["palabras_clave"].astype(str)
    df["texto_ejemplo"] = df["texto_ejemplo"].astype(str)

    df["texto_base_ia"] = df.apply(
        lambda row: (
            row["texto_ejemplo"]
            if row["texto_ejemplo"].strip() and row["texto_ejemplo"].lower() != "nan"
            else row["palabras_clave"]
        ),
        axis=1
    )

    return df


@st.cache_data
def cargar_dataset_documentos():
    if not Path(ARCHIVO_DOCUMENTOS).exists():
        st.warning(f"No se encontró {ARCHIVO_DOCUMENTOS}. Se creará cuando guardes el primer documento.")
        return pd.DataFrame(columns=[
            "documento_id",
            "tipo_documento",
            "titulo_documento",
            "fecha_documento",
            "entidad_origen",
            "area_responsable",
            "categoria",
            "estado",
            "nivel_confidencialidad",
            "prioridad",
            "texto_documento",
            "resumen_documento",
            "palabras_clave",
            "pregunta_ejemplo",
            "respuesta_esperada",
            "fuente_archivo",
            "observaciones"
        ])

    df = pd.read_excel(ARCHIVO_DOCUMENTOS, sheet_name=HOJA_DOCUMENTOS)
    df.columns = df.columns.str.strip()

    columnas_base = [
        "documento_id",
        "tipo_documento",
        "titulo_documento",
        "fecha_documento",
        "entidad_origen",
        "area_responsable",
        "categoria",
        "estado",
        "nivel_confidencialidad",
        "prioridad",
        "texto_documento",
        "resumen_documento",
        "palabras_clave",
        "pregunta_ejemplo",
        "respuesta_esperada",
        "fuente_archivo",
        "observaciones"
    ]

    for col in columnas_base:
        if col not in df.columns:
            df[col] = ""

    df["tipo_documento"] = df["tipo_documento"].astype(str).str.strip()
    df["texto_documento"] = df["texto_documento"].fillna("").astype(str)
    df["titulo_documento"] = df["titulo_documento"].fillna("").astype(str)
    df["resumen_documento"] = df["resumen_documento"].fillna("").astype(str)
    df["palabras_clave"] = df["palabras_clave"].fillna("").astype(str)

    return df[columnas_base]


def cargar_tipos_desde_dataframe(df_tipos):
    tipos = {}

    for _, row in df_tipos.iterrows():
        tipo = row["tipo_documental"]
        palabras = str(row["palabras_clave"]).split(",")
        tipos[tipo] = [p.strip().lower() for p in palabras if p.strip()]

    return tipos


def construir_base_semantica(df_tipos, df_documentos):
    registros = []

    tipos_validos = set(
        df_tipos["tipo_documental"]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )

    # Base 1: catálogo oficial de tipos documentales
    for _, row in df_tipos.iterrows():
        registros.append({
            "tipo_documento": row["tipo_documental"],
            "texto_base": row["texto_base_ia"],
            "origen": "Catálogo oficial de tipos documentales"
        })

    # Base 2: documentos históricos, pero SOLO si el tipo existe en el catálogo oficial.
    # Esto evita que el sistema devuelva tipos no controlados como Reclamo,
    # si Reclamo no forma parte del catálogo oficial de tipificación documental.
    for _, row in df_documentos.iterrows():
        tipo_hist = str(row.get("tipo_documento", "")).strip()

        if tipo_hist not in tipos_validos:
            continue

        texto_historico = " ".join([
            str(row.get("titulo_documento", "")),
            str(row.get("texto_documento", "")),
            str(row.get("resumen_documento", "")),
            str(row.get("palabras_clave", ""))
        ]).strip()

        if texto_historico:
            registros.append({
                "tipo_documento": tipo_hist,
                "texto_base": texto_historico,
                "origen": f"Documento histórico: {row.get('documento_id', '')}"
            })

    return pd.DataFrame(registros)


# =========================================================
# REGLAS FUERTES DE NEGOCIO
# =========================================================

def detectar_tipo_por_reglas_fuertes(texto):
    texto_lower = texto.lower()

    reglas_contrato = [
        "contrato",
        "clausula",
        "cláusula",
        "objeto del contrato",
        "plazo del contrato",
        "resolucion de contrato",
        "resolución de contrato",
        "obligaciones de las partes",
        "el proveedor",
        "el cliente",
        "contraprestación",
        "forma de pago",
        "jurisdiccion",
        "jurisdicción"
    ]

    coincidencias_contrato = sum(
        1 for regla in reglas_contrato
        if regla in texto_lower
    )

    if coincidencias_contrato >= 3:
        return "Contrato", 95, "Regla fuerte contractual"

    return None, 0, None


# =========================================================
# LECTURA DE DOCUMENTOS
# =========================================================

def leer_pdf_texto(file):
    texto = ""
    file.seek(0)

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            contenido = page.extract_text()
            if contenido:
                texto += contenido + "\n"

    return texto


def leer_pdf_ocr(file):
    texto = ""
    file.seek(0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(file.read())
        temp_path = temp_pdf.name

    pdf_document = fitz.open(temp_path)

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(dpi=200)
        img_path = f"temp_page_{page_num}.png"
        pix.save(img_path)

        image = Image.open(img_path)
        texto += pytesseract.image_to_string(image, lang="spa") + "\n"

    return texto


def leer_word(file):
    file.seek(0)
    documento = docx.Document(file)
    return "\n".join([p.text for p in documento.paragraphs])


def leer_txt(file):
    file.seek(0)
    return file.read().decode("utf-8", errors="ignore")


def extraer_texto(file):
    extension = Path(file.name).suffix.lower()

    try:
        if extension == ".pdf":
            texto = leer_pdf_texto(file)

            if texto.strip():
                return texto

            return leer_pdf_ocr(file)

        if extension == ".docx":
            return leer_word(file)

        if extension == ".txt":
            return leer_txt(file)

        return ""

    except Exception as e:
        st.error(f"Error leyendo documento: {str(e)}")
        return ""


# =========================================================
# TIPIFICACIÓN
# =========================================================

def tipificar_por_keywords(texto, tipos_documentales):
    texto_lower = texto.lower()
    puntajes = {}

    for tipo, palabras in tipos_documentales.items():
        puntaje = sum(1 for palabra in palabras if palabra in texto_lower)
        puntajes[tipo] = puntaje

    tipo_sugerido = max(puntajes, key=puntajes.get)
    puntaje_max = puntajes[tipo_sugerido]

    if puntaje_max == 0:
        return "No identificado", 0, puntajes

    confianza = min(95, 50 + puntaje_max * 15)

    return tipo_sugerido, confianza, puntajes


def tipificar_por_ia(texto, df_base_semantica, modelo):
    textos_base = df_base_semantica["texto_base"].fillna("").astype(str).tolist()
    tipos = df_base_semantica["tipo_documento"].fillna("").astype(str).tolist()
    origenes = df_base_semantica["origen"].fillna("").astype(str).tolist()

    embedding_documento = modelo.encode(texto[:5000], convert_to_tensor=True)
    embeddings_base = modelo.encode(textos_base, convert_to_tensor=True)

    similitudes = util.cos_sim(embedding_documento, embeddings_base)[0]

    indice_mejor = int(similitudes.argmax())
    similitud = float(similitudes[indice_mejor])

    tipo_sugerido = tipos[indice_mejor]
    confianza = round(similitud * 100, 2)

    similares = []
    top_indices = similitudes.argsort(descending=True)[:5]

    for idx in top_indices:
        idx = int(idx)
        similares.append({
            "tipo": tipos[idx],
            "similitud": round(float(similitudes[idx]) * 100, 2),
            "referencia": textos_base[idx][:180],
            "origen": origenes[idx]
        })

    return tipo_sugerido, confianza, similares


def tipificar_hibrido(texto, tipos_documentales, df_base_semantica, modelo):
    # 1. Primero aplicamos reglas fuertes de negocio
    tipo_regla, conf_regla, metodo_regla = detectar_tipo_por_reglas_fuertes(texto)

    if tipo_regla and tipo_regla in tipos_documentales:
        similares = []
        return tipo_regla, "Alta", conf_regla, metodo_regla, similares

    # 2. Luego aplicamos palabras clave
    tipo_kw, conf_kw, puntajes = tipificar_por_keywords(texto, tipos_documentales)

    # 3. Luego IA semántica contra catálogo oficial + documentos históricos filtrados
    tipo_ia, conf_ia, similares = tipificar_por_ia(texto, df_base_semantica, modelo)

    # 4. Seguridad: si la IA devuelve un tipo que no existe en el catálogo oficial, se descarta
    if tipo_ia not in tipos_documentales:
        tipo_ia = "No identificado"
        conf_ia = 0

    # 5. Decisión híbrida controlada
    if conf_kw >= 85:
        tipo_final = tipo_kw
        confianza_final = conf_kw
        metodo = "Palabras clave dominantes + validación de catálogo"
    elif conf_ia >= 60 and tipo_ia != "No identificado":
        tipo_final = tipo_ia
        confianza_final = round((conf_ia * 0.60) + (conf_kw * 0.40), 2)
        metodo = "IA semántica + palabras clave + documentos históricos filtrados"
    elif conf_kw > 0:
        tipo_final = tipo_kw
        confianza_final = conf_kw
        metodo = "Palabras clave"
    else:
        tipo_final = "No identificado"
        confianza_final = 0
        metodo = "Sin coincidencia suficiente"

    if confianza_final >= 80:
        nivel = "Alta"
    elif confianza_final >= 60:
        nivel = "Media"
    else:
        nivel = "Baja"

    return tipo_final, nivel, confianza_final, metodo, similares


# =========================================================
# EXTRACCIÓN DE DATOS
# =========================================================

def extraer_datos(texto):
    datos = {}

    fecha = re.search(
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
        texto
    )

    numero = re.search(
        r"\b(N°|Nº|No\.|Número|Num\.?)\s*[:\-]?\s*([A-Za-z0-9\-\/]+)",
        texto,
        re.IGNORECASE
    )

    asunto = re.search(
        r"Asunto\s*[:\-]\s*(.+)",
        texto,
        re.IGNORECASE
    )

    ruc = re.search(
        r"\b\d{11}\b",
        texto
    )

    datos["Fecha"] = fecha.group(1) if fecha else ""
    datos["Número Documento"] = numero.group(2) if numero else ""
    datos["Asunto"] = asunto.group(1).strip() if asunto else ""
    datos["RUC"] = ruc.group(0) if ruc else ""

    return datos


def generar_resumen(texto, max_chars=250):
    texto_limpio = " ".join(texto.split())
    return texto_limpio[:max_chars] + ("..." if len(texto_limpio) > max_chars else "")


def generar_id_documento(df_documentos):
    if df_documentos.empty or "documento_id" not in df_documentos.columns:
        return "DOC-0001"

    numeros = []

    for valor in df_documentos["documento_id"].dropna().astype(str):
        match = re.search(r"(\d+)", valor)
        if match:
            numeros.append(int(match.group(1)))

    siguiente = max(numeros) + 1 if numeros else 1

    return f"DOC-{siguiente:04d}"


def guardar_documento_en_dataset(df_documentos, nuevo_registro):
    df_actualizado = pd.concat(
        [df_documentos, pd.DataFrame([nuevo_registro])],
        ignore_index=True
    )

    # Se guarda principalmente la hoja Dataset_Documentos.
    # Si el archivo ya tiene otras hojas, se mantienen.
    if Path(ARCHIVO_DOCUMENTOS).exists():
        with pd.ExcelWriter(
            ARCHIVO_DOCUMENTOS,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace"
        ) as writer:
            df_actualizado.to_excel(writer, sheet_name=HOJA_DOCUMENTOS, index=False)
    else:
        with pd.ExcelWriter(ARCHIVO_DOCUMENTOS, engine="openpyxl") as writer:
            df_actualizado.to_excel(writer, sheet_name=HOJA_DOCUMENTOS, index=False)

    st.cache_data.clear()

    return df_actualizado


# =========================================================
# UPLOAD
# =========================================================

archivo_documento = st.file_uploader(
    "📄 Subir Documento",
    type=["pdf", "docx", "txt"]
)


# =========================================================
# PROCESAMIENTO
# =========================================================

if archivo_documento:

    df_tipos = cargar_dataset_tipos()
    df_documentos = cargar_dataset_documentos()

    tipos_documentales = cargar_tipos_desde_dataframe(df_tipos)
    df_base_semantica = construir_base_semantica(df_tipos, df_documentos)

    modelo = cargar_modelo_ia()
    texto = extraer_texto(archivo_documento)

    if not texto.strip():
        st.error("No se pudo leer el contenido del documento.")
    else:
        tipo, nivel, confianza, metodo, similares = tipificar_hibrido(
            texto,
            tipos_documentales,
            df_base_semantica,
            modelo
        )

        datos = extraer_datos(texto)

        st.markdown(f"""
        <div class="glass-card">
            <h2>📄 Resultado del Análisis</h2>
            <p><b>Documento:</b> {archivo_documento.name}</p>
            <p><b>Tipo Detectado:</b> {tipo}</p>
            <p><b>Confianza:</b> {confianza}%</p>
            <p><b>Nivel:</b> {nivel}</p>
            <p><b>Método aplicado:</b> {metodo}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("## 🔍 Datos Detectados")

        for campo, valor in datos.items():
            st.markdown(f"""
            <div class="metric-card">
                <b>{campo}</b><br>
                {valor if valor else "No detectado"}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("## 🧠 Documentos / tipos similares")

        for item in similares:
            st.markdown(f"""
            <div class="metric-card">
                <b>{item["tipo"]}</b><br>
                Similitud: {item["similitud"]}%<br>
                Origen: {item["origen"]}<br>
                Referencia: {item["referencia"]}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("## 💾 Guardar resultado en DatasetDocumentos")

        col1, col2 = st.columns(2)

        with col1:
            entidad_origen = st.text_input("Entidad origen", value="")
            area_responsable = st.text_input("Área responsable", value="")
            categoria = st.text_input("Categoría", value="Documental")

        with col2:
            estado = st.selectbox("Estado", ["Pendiente", "Archivado", "Vigente", "En revisión"])
            nivel_confidencialidad = st.selectbox("Nivel confidencialidad", ["Interno", "Público", "Reservado", "Confidencial"])
            prioridad = st.selectbox("Prioridad", ["Baja", "Media", "Alta"])

        titulo_sugerido = datos["Asunto"] if datos["Asunto"] else Path(archivo_documento.name).stem
        titulo_documento = st.text_input("Título documento", value=titulo_sugerido)

        observaciones = st.text_area(
            "Observaciones",
            value=f"Tipificado automáticamente con método: {metodo}. Confianza: {confianza}%."
        )

        if st.button("💾 Guardar documento tipificado"):

            documento_id = generar_id_documento(df_documentos)

            nuevo_registro = {
                "documento_id": documento_id,
                "tipo_documento": tipo,
                "titulo_documento": titulo_documento,
                "fecha_documento": datos["Fecha"] if datos["Fecha"] else datetime.now().strftime("%Y-%m-%d"),
                "entidad_origen": entidad_origen,
                "area_responsable": area_responsable,
                "categoria": categoria,
                "estado": estado,
                "nivel_confidencialidad": nivel_confidencialidad,
                "prioridad": prioridad,
                "texto_documento": texto,
                "resumen_documento": generar_resumen(texto),
                "palabras_clave": ", ".join([item["tipo"] for item in similares[:3]]),
                "pregunta_ejemplo": f"¿Qué indica el documento {documento_id}?",
                "respuesta_esperada": generar_resumen(texto, max_chars=350),
                "fuente_archivo": archivo_documento.name,
                "observaciones": observaciones
            }

            guardar_documento_en_dataset(df_documentos, nuevo_registro)

            st.success(f"Documento guardado correctamente en {ARCHIVO_DOCUMENTOS} con ID {documento_id}.")

            with open(ARCHIVO_DOCUMENTOS, "rb") as f:
                st.download_button(
                    label="⬇️ Descargar DatasetDocumentos actualizado",
                    data=f,
                    file_name=ARCHIVO_DOCUMENTOS,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

else:
    st.info("Ahora sube el documento a analizar.")
