"""
=============================================================================
TRADUCTOR DE PAPERS Y DOCUMENTOS ACADÉMICOS (PDF a DOCX)
Streamlit + Google Gemini + LangChain + PyMuPDF + python-docx
=============================================================================
"""

import io
import re
import zipfile
import time
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
import json

# Número máximo de solicitudes simultáneas a Gemini.
# Se mantiene controlado para evitar saturar la API.
MAX_CONCURRENT_REQUESTS = 3

# Configuración de reintentos para errores de cuota/rate limit de Gemini.
MAX_RETRIES = 3

# Tiempo de espera de seguridad si Gemini no proporciona
# correctamente el retry_delay.
DEFAULT_RETRY_DELAY = 60

# Librerías para extracción de PDF
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

# Librerías para generación de DOCX
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

# Google Gemini API — nuevo SDK
from google import genai

# LangChain y Agentes
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.prebuilt import create_react_agent
    from langchain_core.tools import tool
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


# =============================================================================
# 1. EXTRACCIÓN DE TEXTO RESPETANDO COLUMNAS (PyMuPDF / pdfplumber)
# =============================================================================

def extract_text_two_columns_pymupdf(file_bytes: bytes) -> Tuple[str, str, bool]:
    """
    Extrae texto de un PDF preservando el orden de lectura de dos columnas.
    Lee primero la columna izquierda completa antes de la derecha.
    
    Retorna:
        - full_text: Texto completo ordenado.
        - paper_title: Título detectado o estimado.
        - is_scanned: True si el documento no tiene capa de texto digital.
    """
    if fitz is None:
        raise ImportError("PyMuPDF (fitz) no está instalado.")

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    all_pages_text = []
    detected_title = ""
    total_char_count = 0

    for page_idx, page in enumerate(doc):
        rect = page.rect
        page_width = rect.width
        mid_x = page_width / 2.0

        # Obtener bloques de texto: (x0, y0, x1, y1, text, block_no, block_type)
        # block_type == 0 es texto
        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]

        if not text_blocks:
            continue

        # Intentar extraer el título del paper en la primera página
        if page_idx == 0 and not detected_title:
            for b in text_blocks[:3]:
                candidate = b[4].strip().replace('\n', ' ')
                # Un título suele tener longitud moderada y estar al inicio
                if 15 <= len(candidate) <= 200 and not candidate.lower().startswith("issn"):
                    detected_title = candidate
                    break

        # Clasificación de bloques por columnas
        # Bloques anchos (título, resumen, cabecera que cruzan el ancho)
        top_spanning = []
        left_column = []
        right_column = []
        bottom_spanning = []

        for b in text_blocks:
            x0, y0, x1, y1, text, _, _ = b
            block_width = x1 - x0
            
            # Si el bloque cruza más del 65% del ancho de la página, es un bloque de ancho completo
            if block_width > (page_width * 0.65):
                if y0 < rect.height * 0.35:
                    top_spanning.append(b)
                else:
                    bottom_spanning.append(b)
            # Columna izquierda
            elif x0 < mid_x and x1 <= (mid_x + 40):
                left_column.append(b)
            # Columna derecha
            elif x0 >= (mid_x - 40):
                right_column.append(b)
            else:
                # Si cae en medio, clasificar por el centro del bloque
                block_center_x = (x0 + x1) / 2.0
                if block_center_x < mid_x:
                    left_column.append(b)
                else:
                    right_column.append(b)

        # Ordenar cada grupo por posición vertical (y0)
        top_spanning.sort(key=lambda b: b[1])
        left_column.sort(key=lambda b: b[1])
        right_column.sort(key=lambda b: b[1])
        bottom_spanning.sort(key=lambda b: b[1])

        # Ensamblar en orden de lectura natural:
        # 1. Cabecera/Título ancho -> 2. Columna izquierda completa -> 3. Columna derecha completa -> 4. Pie ancho
        page_ordered_blocks = top_spanning + left_column + right_column + bottom_spanning

        page_text_pieces = []
        for b in page_ordered_blocks:
            clean_block_text = b[4].strip()
            if clean_block_text:
                page_text_pieces.append(clean_block_text)
                total_char_count += len(clean_block_text)

        if page_text_pieces:
            all_pages_text.append("\n\n".join(page_text_pieces))

    doc.close()

    full_text = "\n\n".join(all_pages_text)
    is_scanned = total_char_count < 60  # Menos de 60 caracteres indica PDF escaneado/imagen

    return full_text, detected_title, is_scanned


def extract_text_fallback_pdfplumber(file_bytes: bytes) -> Tuple[str, str, bool]:
    """
    Extracción alternativa con pdfplumber si PyMuPDF no está disponible.
    """
    if pdfplumber is None:
        raise ImportError("pdfplumber no está instalado.")

    full_text_list = []
    total_chars = 0
    title = ""

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for idx, page in enumerate(pdf.pages):
            text = page.extract_text(layout=True) or ""
            text = text.strip()
            if text:
                full_text_list.append(text)
                total_chars += len(text)
                if idx == 0 and not title:
                    lines = [l.strip() for l in text.split('\n') if l.strip()]
                    if lines:
                        title = lines[0]

    full_text = "\n\n".join(full_text_list)
    is_scanned = total_chars < 60
    return full_text, title, is_scanned


def extract_pdf_content(file_bytes: bytes, filename: str) -> Tuple[str, str, bool]:
    """
    Función principal de extracción con manejo de fallback.
    """
    if fitz is not None:
        return extract_text_two_columns_pymupdf(file_bytes)
    elif pdfplumber is not None:
        return extract_text_fallback_pdfplumber(file_bytes)
    else:
        raise RuntimeError("No se encontró ningún motor de extracción PDF (PyMuPDF o pdfplumber).")


# =============================================================================
# 2. SEGMENTACIÓN INTELIGENTE (Párrafos y Frases sin corte abrupto)
# =============================================================================

def segment_text_into_chunks(
    text: str,
    max_chars: int = 4000,
    target_chunks: Optional[int] = None
) -> List[str]:
    """
    Divide el texto en fragmentos de hasta ~4000 caracteres,
    respetando párrafos y oraciones completas siempre que sea posible.

    Si un párrafo supera el tamaño máximo, se divide por oraciones
    para evitar cortes abruptos en medio de una idea.

    Args:
        text: Texto completo extraído del PDF.
        max_chars: Número máximo aproximado de caracteres por fragmento.

    Returns:
        Lista de fragmentos de texto listos para ser traducidos.
    """
    if not text or not text.strip():
        return []

    # Normalizar saltos de línea
    normalized_text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Separar por párrafos
    raw_paragraphs = [
        p.strip()
        for p in normalized_text.split('\n\n')
        if p.strip()
    ]

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0

    for paragraph in raw_paragraphs:
        p_len = len(paragraph)

        # -------------------------------------------------------------
        # CASO 1: El párrafo cabe dentro del chunk actual
        # -------------------------------------------------------------
        if current_length + p_len + 2 <= max_chars:

            current_chunk.append(paragraph)
            current_length += p_len + 2

        else:

            # ---------------------------------------------------------
            # CASO 2: El párrafo supera por sí solo el tamaño máximo
            # ---------------------------------------------------------
            if p_len > max_chars:

                # Guardar primero el chunk acumulado
                if current_chunk:
                    chunks.append(
                        "\n\n".join(current_chunk)
                    )

                    current_chunk = []
                    current_length = 0

                # Dividir el párrafo por oraciones completas
                sentences = re.split(
                    r'(?<=[.!?])\s+',
                    paragraph
                )

                temp_sentence_chunk: List[str] = []
                temp_len = 0

                for sentence in sentences:

                    sentence_clean = sentence.strip()

                    if not sentence_clean:
                        continue

                    sentence_len = len(sentence_clean)

                    # La oración todavía cabe
                    if temp_len + sentence_len + 1 <= max_chars:

                        temp_sentence_chunk.append(
                            sentence_clean
                        )

                        temp_len += sentence_len + 1

                    else:

                        # Guardar el grupo de oraciones anterior
                        if temp_sentence_chunk:
                            chunks.append(
                                " ".join(temp_sentence_chunk)
                            )

                        # Comenzar nuevo fragmento
                        temp_sentence_chunk = [
                            sentence_clean
                        ]

                        temp_len = sentence_len

                # Guardar las últimas oraciones pendientes
                if temp_sentence_chunk:
                    chunks.append(
                        " ".join(temp_sentence_chunk)
                    )

            # ---------------------------------------------------------
            # CASO 3: El párrafo cabe en un chunk nuevo,
            # pero no en el chunk actual
            # ---------------------------------------------------------
            else:

                # Guardar el chunk actual
                if current_chunk:
                    chunks.append(
                        "\n\n".join(current_chunk)
                    )

                # Comenzar un nuevo chunk con este párrafo
                current_chunk = [paragraph]
                current_length = p_len

    # -------------------------------------------------------------
    # Guardar el último chunk pendiente
    # -------------------------------------------------------------
    if current_chunk:
        chunks.append(
            "\n\n".join(current_chunk)
        )

    return chunks


# =============================================================================
# 3. TRADUCCIÓN: MODO DIRECTO (Google Generative AI)
# =============================================================================

def build_translation_prompt(
    chunk: str,
    target_language: str
) -> str:
    """
    Construye el prompt estándar utilizado tanto por el
    modo Directo como por el modo Batch.
    """

    system_instruction = (
        "Eres un traductor académico profesional y riguroso, "
        "especializado en papers científicos y publicaciones universitarias.\n"
        f"Tu tarea es traducir el texto recibido al idioma: {target_language}.\n\n"

        "REGLAS OBLIGATORIAS:\n"
        "1. Devuelve ÚNICAMENTE la traducción limpia del texto. "
        "No agregues preámbulos, notas del traductor, saludos, "
        "advertencias ni explicaciones adicionales.\n"

        "2. Mantén la terminología técnica y el tono formal académico.\n"

        "3. NO traduzcas fórmulas matemáticas, ecuaciones, variables "
        "ni fragmentos de código.\n"

        "4. NO traduzcas referencias bibliográficas ni claves de "
        "citación estándar (ej. [1], (Smith et al., 2021)).\n"

        "5. NO traduzcas nombres propios de autores, nombres de "
        "universidades ni afiliaciones institucionales.\n"

        "6. Preserva los saltos de línea y la estructura de "
        "párrafos original del texto.\n\n"

        "Texto a traducir:\n\n"
        f"{chunk}"
    )

    return system_instruction

def translate_chunk_direct(
    chunk: str,
    target_language: str,
    client,
    model_name: str
) -> str:
    """
    Traduce un fragmento individual utilizando Gemini.
    """

    prompt = build_translation_prompt(
        chunk=chunk,
        target_language=target_language
    )

    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )

    if response and response.text:
        return response.text.strip()

    return chunk


def translate_chunks_concurrent(
    chunks: List[str],
    target_language: str,
    client,
    model_name: str,
    max_workers: int = MAX_CONCURRENT_REQUESTS,
    progress_callback=None,
    live_callback=None
) -> List[str]:
    """
    Traduce múltiples fragmentos de forma concurrente utilizando
    un número controlado de trabajadores.

    Los resultados se mantienen en el mismo orden que los chunks originales.

    Args:
        chunks: Lista de fragmentos a traducir.
        target_language: Idioma de destino.
        api_key: API Key de Gemini.
        model_name: Modelo de Gemini.
        max_workers: Número máximo de solicitudes simultáneas.
        progress_callback: Función opcional para reportar progreso.
        live_callback: Función opcional para mostrar resultados.

    Returns:
        Lista de traducciones en el mismo orden de entrada.
    """

    if not chunks:
        return []

    # Reservar espacio para mantener el orden original.
    translated_chunks = [None] * len(chunks)

    # Número de fragmentos que ya terminaron.
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        # Crear todas las tareas.
        future_to_index = {
            executor.submit(
                translate_chunk_with_retry,
                chunk=chunk,
                target_language=target_language,
                client=client,
                model_name=model_name
            ): index
            for index, chunk in enumerate(chunks)
        }

        # Procesar resultados conforme van terminando.
        for future in as_completed(future_to_index):

            chunk_index = future_to_index[future]

            try:
                result = future.result()

                # Guardar en su posición original.
                translated_chunks[chunk_index] = result

            except Exception as e:

                translated_chunks[chunk_index] = chunks[chunk_index]
            
                st.error(
                    f"❌ Error definitivo traduciendo el fragmento "
                    f"{chunk_index + 1}: {str(e)}"
                )

            completed += 1

            # Informar progreso.
            if progress_callback:
                progress_callback(
                    completed,
                    len(chunks),
                    chunk_index,
                    translated_chunks[chunk_index]
                )

    return translated_chunks

def get_retry_delay_from_error(error: Exception) -> int:
    """
    Extrae el tiempo de espera recomendado por Gemini desde
    el mensaje del error 429.

    Gemini puede devolver mensajes como:

        Please retry in 59.509833912s.

    o:

        retry_delay { seconds: 59 }

    Si no se encuentra ninguno, se utiliza DEFAULT_RETRY_DELAY.
    """

    error_message = str(error)

    # Intentar obtener:
    # "Please retry in 59.509833912s"
    match = re.search(
        r"Please retry in\s+([\d.]+)s",
        error_message,
        re.IGNORECASE
    )

    if match:
        return max(1, int(float(match.group(1))) + 1)

    # Intentar obtener:
    # "retry_delay { seconds: 59 }"
    match = re.search(
        r"retry_delay\s*\{\s*seconds:\s*(\d+)",
        error_message,
        re.IGNORECASE
    )

    if match:
        return max(1, int(match.group(1)) + 1)

    # Si Gemini no proporciona el tiempo, utilizar
    # el valor de seguridad.
    return DEFAULT_RETRY_DELAY

def translate_chunk_with_retry(
    chunk: str,
    target_language: str,
    client,
    model_name: str,
    max_retries: int = MAX_RETRIES
) -> str:
    """
    Traduce un chunk y reintenta automáticamente ante errores
    temporales de Gemini como 429 y 503.
    """

    attempt = 0

    while True:

        try:
            return translate_chunk_direct(
                chunk=chunk,
                target_language=target_language,
                client=client,
                model_name=model_name
            )

        except Exception as e:

            error_message = str(e)
            error_lower = error_message.lower()

            is_temporary_error = (
                "429" in error_message
                or "503" in error_message
                or "quota exceeded" in error_lower
                or "rate limit" in error_lower
                or "unavailable" in error_lower
                or "high demand" in error_lower
                or "temporarily" in error_lower
            )

            if not is_temporary_error:
                raise

            if attempt >= max_retries:
                raise RuntimeError(
                    f"Gemini no pudo procesar el fragmento después de "
                    f"{max_retries} reintentos. "
                    f"Último error: {error_message}"
                ) from e

            # Primero intentamos obtener el retry_delay
            retry_delay = get_retry_delay_from_error(e)

            # Para 503 normalmente Gemini no proporciona retry_delay.
            # Aplicamos backoff progresivo.
            if "503" in error_message:
                retry_delay = min(
                    15 * (2 ** attempt),
                    60
                )

            attempt += 1

            st.warning(
                f"⏳ Gemini respondió temporalmente con un error "
                f"({('503' if '503' in error_message else '429')}). "
                f"Reintentando en {retry_delay} segundos "
                f"(intento {attempt}/{max_retries})..."
            )

            time.sleep(retry_delay)

# =============================================================================
# 4. TRADUCCIÓN: MODO AGENTE (LangChain + Gemini + Tools)
# =============================================================================

# Mapeo de códigos ISO a nombres de idiomas
LANG_CODE_MAP = {
    "español": "es",
    "spanish": "es",
    "inglés": "en",
    "english": "en",
    "francés": "fr",
    "french": "fr",
    "portugués": "pt",
    "portuguese": "pt",
    "alemán": "de",
    "german": "de",
    "italiano": "it",
    "italian": "it",
}


def normalize_language_code(language: str) -> str:
    """
    Convierte el nombre del idioma a su código ISO.
    
    Ejemplos:
        Español -> es
        Inglés  -> en
        Francés -> fr
    """

    if not language:
        return "unknown"

    normalized = language.strip().lower()

    return LANG_CODE_MAP.get(
        normalized,
        normalized[:2]
    )


def create_translation_agent_executor(
    api_key: str,
    target_language: str,
    model_name: str = "gemini-3.6-flash"
):
    """
    Crea el agente supervisor de traducción.

    IMPORTANTE:
    El agente NO implementa un motor de traducción independiente.

    El agente únicamente:
        1. Detecta el idioma.
        2. Decide si es necesario traducir.
        3. Utiliza translate_academic_text_tool.

    El motor real de traducción es el mismo utilizado
    por el Modo Directo.
    """

    if not LANGCHAIN_AVAILABLE:
        raise RuntimeError(
            "Las dependencias de LangChain/LangGraph no están disponibles."
        )

    # -------------------------------------------------------------------------
    # Motor Gemini utilizado por las herramientas del agente
    # -------------------------------------------------------------------------

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.1,
    )

    # -------------------------------------------------------------------------
    # Herramienta 1: detección de idioma
    # -------------------------------------------------------------------------

    @tool
    def detect_language_tool(text: str) -> str:
        """
        Detecta el idioma predominante del fragmento.

        Devuelve un código ISO como:
            es = español
            en = inglés
            fr = francés
            pt = portugués
            de = alemán
            it = italiano
        """

        try:

            sample = text[:1000].strip()

            if not sample:
                return "unknown"

            lang_code = detect(sample)

            return lang_code

        except Exception:
            return "unknown"

    # -------------------------------------------------------------------------
    # Herramienta 2: traducción académica
    # -------------------------------------------------------------------------

    @tool
    def translate_academic_text_tool(
        text: str,
        target_lang: str
    ) -> str:
        """
        Traduce un fragmento académico utilizando el mismo LLM
        configurado para el agente.
        """

        if not text or not text.strip():
            return text

        translation_prompt = (
            "Eres un traductor académico profesional especializado "
            "en papers científicos y publicaciones universitarias.\n\n"

            f"Traduce el siguiente texto al idioma {target_lang}.\n\n"

            "REGLAS OBLIGATORIAS:\n"
            "1. Devuelve ÚNICAMENTE la traducción.\n"
            "2. No agregues explicaciones, comentarios, etiquetas "
            "ni preámbulos.\n"
            "3. Mantén la terminología técnica y el tono académico.\n"
            "4. NO traduzcas fórmulas matemáticas, ecuaciones, "
            "variables ni código.\n"
            "5. NO traduzcas referencias bibliográficas ni claves "
            "de citación estándar como [1] o (Smith et al., 2021).\n"
            "6. NO traduzcas nombres propios de autores, universidades "
            "ni afiliaciones institucionales.\n"
            "7. Conserva la estructura de párrafos y saltos de línea "
            "cuando sea posible.\n\n"

            "TEXTO A TRADUCIR:\n\n"
            f"{text}"
        )

        response = llm.invoke(translation_prompt)

        if hasattr(response, "content"):
            return str(response.content).strip()

        return str(response).strip()

    # -------------------------------------------------------------------------
    # Herramientas disponibles para el agente
    # -------------------------------------------------------------------------

    tools = [
        detect_language_tool,
        translate_academic_text_tool
    ]

    # -------------------------------------------------------------------------
    # Instrucciones del agente
    # -------------------------------------------------------------------------

    system_message = (
        "Eres un agente supervisor de traducción académica.\n\n"

        f"El idioma objetivo es: {target_language}.\n"
        f"El código ISO del idioma objetivo es: "
        f"{normalize_language_code(target_language)}.\n\n"

        "FLUJO OBLIGATORIO:\n"

        "1. Utiliza detect_language_tool para identificar "
        "el idioma del fragmento.\n\n"

        "2. Si el idioma detectado coincide con el idioma objetivo, "
        "devuelve el fragmento exactamente igual.\n\n"

        "3. Si el idioma detectado es diferente al idioma objetivo, "
        "utiliza translate_academic_text_tool.\n\n"

        "4. No inventes contenido.\n\n"

        "5. No resumas el texto.\n\n"

        "6. No elimines información.\n\n"

        "7. La respuesta final debe contener EXCLUSIVAMENTE "
        "el texto traducido o el texto original cuando no sea "
        "necesaria la traducción."
    )

    # -------------------------------------------------------------------------
    # Compatibilidad con versiones actuales de LangGraph
    # -------------------------------------------------------------------------

    try:

        # Las versiones actuales de LangGraph utilizan `prompt`
        # en lugar de `state_modifier`.

        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=system_message
        )

    except TypeError:

        # Compatibilidad adicional con algunas versiones
        # que utilizan el primer argumento posicional.

        agent = create_react_agent(
            llm,
            tools,
            prompt=system_message
        )

    return agent


def translate_chunk_agent(
    agent_executor,
    chunk: str,
    target_language: str
) -> str:
    """
    Procesa un chunk mediante el agente.

    El agente decide si debe conservarlo o traducirlo.
    """

    if not chunk or not chunk.strip():
        return chunk

    try:

        target_code = normalize_language_code(target_language)

        user_message = (
            f"Procesa el siguiente fragmento académico.\n\n"
            f"Idioma objetivo: {target_language} ({target_code}).\n\n"
            "Debes detectar primero el idioma y posteriormente "
            "traducir únicamente si es necesario.\n\n"
            "FRAGMENTO:\n\n"
            f"{chunk}"
        )

        result = agent_executor.invoke(
            {
                "messages": [
                    (
                        "human",
                        user_message
                    )
                ]
            }
        )

        messages = result.get("messages", [])

        if not messages:
            return chunk

        # Obtener el último mensaje generado por el agente.
        output = messages[-1]

        if hasattr(output, "content"):
            content = output.content
        else:
            content = str(output)

        if not content:
            return chunk

        return str(content).strip()

    except Exception as e:

        st.warning(
            "⚠️ El agente no pudo procesar el fragmento. "
            f"Se conservará el texto original. Detalle: {str(e)}"
        )

        return chunk


def translate_chunks_agent_concurrent(
    chunks: List[str],
    target_language: str,
    agent_executor,
    max_workers: int = MAX_CONCURRENT_REQUESTS,
    progress_callback=None
) -> List[str]:
    """
    Traduce los chunks mediante el agente manteniendo
    el orden original.

    NOTA:
    Se mantiene una función separada del motor directo para
    que ambos modos puedan evolucionar independientemente.
    """

    if not chunks:
        return []

    translated_chunks = [None] * len(chunks)

    completed = 0

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        future_to_index = {
            executor.submit(
                translate_chunk_agent,
                agent_executor,
                chunk,
                target_language
            ): index
            for index, chunk in enumerate(chunks)
        }

        for future in as_completed(future_to_index):

            chunk_index = future_to_index[future]

            try:

                result = future.result()

                translated_chunks[chunk_index] = result

            except Exception as e:

                translated_chunks[chunk_index] = chunks[chunk_index]

                st.error(
                    f"❌ Error procesando el fragmento "
                    f"{chunk_index + 1} mediante el agente: {str(e)}"
                )

            completed += 1

            if progress_callback:

                progress_callback(
                    completed,
                    len(chunks),
                    chunk_index,
                    translated_chunks[chunk_index]
                )

    return translated_chunks


# =============================================================================
# 5. RECONSTRUCCIÓN DEL DOCUMENTO (.docx)
# =============================================================================

def generate_docx(title: str, text_content: str, source_filename: str = "") -> io.BytesIO:
    """
    Genera un documento .docx limpio en flujo normal de una columna,
    con formato tipográfico académico elegante y profesional.
    """
    doc = Document()

    # Configuración de márgenes estándar (1 pulgada / 2.54 cm)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Configuración de estilo normal
    style_normal = doc.styles['Normal']
    font = style_normal.font
    font.name = 'Calibri'
    font.size = Pt(11)
    font.color.rgb = RGBColor(0x22, 0x22, 0x22)

    # Título principal del documento
    display_title = title.strip() if title.strip() else (f"Traducción: {source_filename}" if source_filename else "Documento Traducido")
    title_heading = doc.add_heading(display_title, level=1)
    title_heading.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in title_heading.runs:
        run.font.name = 'Calibri'
        run.font.size = Pt(18)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x11, 0x2A, 0x46)  # Azul académico elegante

    # Subtítulo con metadatos
    meta_p = doc.add_paragraph()
    meta_p.paragraph_format.space_after = Pt(18)
    meta_p.paragraph_format.line_spacing = 1.15
    meta_run = meta_p.add_run(f"Documento traducido automáticamente | Archivo original: {source_filename or 'PDF'}")
    meta_run.font.size = Pt(9.5)
    meta_run.font.italic = True
    meta_run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    # Separador sutil
    doc.add_paragraph("―" * 40).paragraph_format.space_after = Pt(12)

    # Agregar párrafos de texto traducido
    paragraphs = [p.strip() for p in text_content.split('\n\n') if p.strip()]

    for p_text in paragraphs:
        # Detectar si un párrafo parece un subtítulo corto (ej: "1. Introducción", "Abstract", "Metodología")
        if len(p_text) < 80 and ('\n' not in p_text) and (
            p_text.isupper() or 
            re.match(r'^(?:[0-9]+\.|\bAbstract\b|\bResumen\b|\bIntroducción\b|\bMethod\b|\bResultados\b|\bConclusiones\b|\bReferences\b|\bBibliografía\b)', p_text, re.IGNORECASE)
        ):
            h = doc.add_heading(p_text, level=2)
            for r in h.runs:
                r.font.name = 'Calibri'
                r.font.size = Pt(13)
                r.font.bold = True
                r.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
            h.paragraph_format.space_before = Pt(12)
            h.paragraph_format.space_after = Pt(4)
        else:
            p = doc.add_paragraph()
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.add_run(p_text)

    # Guardar en memoria BytesIO
    docx_io = io.BytesIO()
    doc.save(docx_io)
    docx_io.seek(0)
    return docx_io

def generate_txt(text_content: str) -> bytes:
    """
    Genera un archivo TXT con el contenido traducido.

    El contenido se guarda en UTF-8 para preservar correctamente
    caracteres especiales, tildes, símbolos y otros caracteres
    utilizados en textos académicos.
    """
    if not text_content:
        text_content = ""

    return text_content.encode("utf-8")

def generate_txt_content(text_content: str) -> bytes:
    """
    Genera el contenido de un archivo TXT a partir del texto traducido.

    Se utiliza UTF-8 para conservar correctamente:
    - tildes
    - ñ
    - caracteres científicos
    - símbolos
    - caracteres de otros idiomas
    """

    if not text_content:
        text_content = ""

    return text_content.encode("utf-8")

def generate_zip_package(translated_files_data: Dict[str, bytes]) -> io.BytesIO:
    """
    Empaqueta múltiples archivos .docx generados en un archivo .zip en memoria.
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for filename, file_bytes in translated_files_data.items():
            zip_file.writestr(filename, file_bytes)
    zip_buffer.seek(0)
    return zip_buffer
    
# =============================================================================
# 6. TRADUCCIÓN: MODO BATCH
# =============================================================================

def create_batch_jsonl(
    chunks: List[str],
    target_language: str,
    output_path: str
) -> str:
    """
    Genera un archivo JSONL compatible con Gemini Batch API.

    Cada línea representa una solicitud independiente.
    """

    with open(output_path, "w", encoding="utf-8") as f:

        for index, chunk in enumerate(chunks):

            request = {
                "key": f"chunk-{index}",
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "text": build_translation_prompt(
                                        chunk=chunk,
                                        target_language=target_language
                                    )
                                }
                            ]
                        }
                    ]
                }
            }

            f.write(
                json.dumps(
                    request,
                    ensure_ascii=False
                ) + "\n"
            )

    return output_path

def create_translation_batch(
    chunks: List[str],
    target_language: str,
    client,
    model_name: str
):
    """
    Crea un trabajo Batch de Gemini para todos los chunks.
    """

    import tempfile
    import os

    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".jsonl",
        delete=False,
        encoding="utf-8"
    )

    temp_file.close()

    jsonl_path = create_batch_jsonl(
        chunks=chunks,
        target_language=target_language,
        output_path=temp_file.name
    )

    try:

        uploaded_file = client.files.upload(
            file=jsonl_path
        )

        batch_job = client.batches.create(
            model=model_name,
            src=uploaded_file.name,
            config={
                "display_name": "paper-translation-batch"
            }
        )

        return batch_job

    finally:

        try:
            os.remove(jsonl_path)
        except OSError:
            pass

def wait_for_batch_completion(
    client,
    batch_job,
    poll_interval: int = 10
):
    """
    Espera hasta que Gemini termine el procesamiento Batch.
    """

    while True:

        batch_job = client.batches.get(
            name=batch_job.name
        )

        state = batch_job.state.name

        if state == "JOB_STATE_SUCCEEDED":
            return batch_job

        if state == "JOB_STATE_FAILED":
            raise RuntimeError(
                f"El Batch de Gemini falló: "
                f"{getattr(batch_job, 'error', 'Error desconocido')}"
            )

        if state == "JOB_STATE_CANCELLED":
            raise RuntimeError(
                "El Batch de Gemini fue cancelado."
            )

        if state == "JOB_STATE_EXPIRED":
            raise RuntimeError(
                "El Batch de Gemini expiró después de 48 horas."
            )

        time.sleep(poll_interval)

def extract_batch_results(
    client,
    batch_job,
    chunks: List[str]
) -> List[str]:
    """
    Extrae las traducciones generadas por Gemini Batch API
    y las devuelve en el mismo orden de los chunks originales.
    """

    if not batch_job.dest:
        raise RuntimeError(
            "El Batch terminó correctamente pero no contiene destino."
        )

    result_file_name = batch_job.dest.file_name

    if not result_file_name:
        raise RuntimeError(
            "No se encontró el archivo de resultados del Batch."
        )

    file_content = client.files.download(
        file=result_file_name
    )

    if isinstance(file_content, bytes):
        content = file_content.decode(
            "utf-8",
            errors="replace"
        )
    else:
        content = str(file_content)

    translated_chunks = [None] * len(chunks)

    for line_number, line in enumerate(
        content.splitlines()
    ):

        if not line.strip():
            continue

        try:

            result = json.loads(line)

            key = result.get("key", "")

            match = re.search(
                r"chunk-(\d+)",
                key
            )

            if not match:
                continue

            chunk_index = int(
                match.group(1)
            )

            response = result.get(
                "response"
            )

            if not response:
                translated_chunks[chunk_index] = chunks[chunk_index]
                continue

            candidates = response.get(
                "candidates",
                []
            )

            if not candidates:
                translated_chunks[chunk_index] = chunks[chunk_index]
                continue

            parts = (
                candidates[0]
                .get("content", {})
                .get("parts", [])
            )

            translated_text = "".join(
                part.get("text", "")
                for part in parts
                if isinstance(part, dict)
            ).strip()

            if translated_text:
                translated_chunks[chunk_index] = translated_text
            else:
                translated_chunks[chunk_index] = chunks[chunk_index]

        except Exception as e:

            st.warning(
                f"⚠️ No se pudo procesar "
                f"la línea {line_number + 1} "
                f"del resultado Batch: {str(e)}"
            )

    # Protección final: ningún chunk queda como None
    for index in range(len(translated_chunks)):

        if not translated_chunks[index]:
            translated_chunks[index] = chunks[index]

    return translated_chunks

def translate_chunks_batch(
    chunks: List[str],
    target_language: str,
    client,
    model_name: str,
    progress_callback=None
) -> List[str]:
    """
    Traduce todos los chunks utilizando Gemini Batch API.
    """

    if not chunks:
        return []

    st.info(
        f"📦 Enviando {len(chunks)} fragmentos "
        f"a Gemini Batch API..."
    )

    batch_job = create_translation_batch(
        chunks=chunks,
        target_language=target_language,
        client=client,
        model_name=model_name
    )

    st.success(
        f"✅ Batch creado correctamente: "
        f"`{batch_job.name}`"
    )

    st.info(
        "⏳ Gemini está procesando el lote. "
        "Este proceso es asíncrono y puede tardar."
    )

    completed_job = wait_for_batch_completion(
        client=client,
        batch_job=batch_job,
        poll_interval=10
    )

    st.success(
        "🎉 Gemini terminó de procesar el Batch."
    )

    translated_chunks = extract_batch_results(
        client=client,
        batch_job=completed_job,
        chunks=chunks
    )

    if progress_callback:

        for index, translation in enumerate(
            translated_chunks
        ):

            progress_callback(
                index + 1,
                len(translated_chunks),
                index,
                translation
            )

    return translated_chunks



# =============================================================================
# 7. INTERFAZ STREAMLIT
# =============================================================================

def init_session_state():
    """Inicializa variables en session_state para persistencia de estado."""
    if "translated_docs" not in st.session_state:
        st.session_state.translated_docs = {}  # {filename: {"text": str, "title": str, "docx_bytes": bytes}}
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False


def main():
    st.set_page_config(
        page_title="Traductor de Papers Académicos",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_session_state()

    # Encabezado Principal
    st.title("📄 Traductor de Papers y Documentos Académicos")
    st.markdown(
        "Traduce artículos científicos, informes y papers PDF a formato Word (.docx) "
        "respetando el flujo de lectura en dos columnas, fórmulas y citas bibliográficas."
    )

    # -------------------------------------------------------------------------
    # BARRA LATERAL: CONFIGURACIÓN
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Configuración")

        # 1. API Key de Google Gemini (Campo protegido)
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            placeholder="AIzaSy...",
            help="Ingresa tu clave de API de Google AI Studio. No queda almacenada en disco."
        )

        st.divider()

        # 2. Selector de Idioma Destino
        target_language_options = [
            "Español",
            "Inglés",
            "Francés",
            "Portugués",
            "Alemán",
            "Italiano"
        ]
        selected_language = st.selectbox(
            "🌐 Idioma de destino",
            options=target_language_options,
            index=0
        )

        st.divider()

        # 3. Selección de Modo de Traducción
        mode_options = ["Modo Directo (Gemini API)"]
        mode_options = [
            "Modo Directo (Gemini API)",
            "Modo Batch (Gemini Batch API)"
        ]
        
        if LANGCHAIN_AVAILABLE:
            mode_options.append(
                "Modo Agente (LangChain + Gemini)"
            )
        translation_mode = st.radio(
            "🧠 Modo de Traducción",
            options=mode_options,
            index=0,
            help=(
                "• Modo Directo: Traducción rápida y precisa con prompt de especialidad académica.\n"
                + ("• Modo Agente: Agente autónomo con herramientas de detección de idioma y traducción selectiva." if LANGCHAIN_AVAILABLE else "")
            )
        )

        # 4. Modelo de Gemini
        model_choice = st.selectbox(
            "🤖 Modelo Gemini",
            options=["gemini-3.6-flash", "gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
            index=0
        )

        # 5. Parámetros de segmentación
        with st.expander("Ajustes avanzados de segmentación"):
        
            # ---------------------------------------------------------
            # Tamaño máximo de cada chunk
            # ---------------------------------------------------------
            chunk_size = st.slider(
                "📏 Tamaño máx. por fragmento (caracteres)",
                min_value=2000,
                max_value=5000,
                value=4000,
                step=250,
                help=(
                    "Establece el tamaño máximo permitido para cada fragmento. "
                    "Este límite siempre tiene prioridad para evitar enviar "
                    "fragmentos demasiado grandes a Gemini."
                )
            )
        
            # ---------------------------------------------------------
            # Cantidad objetivo de chunks
            # ---------------------------------------------------------
            chunk_mode = st.radio(
                "📦 Cantidad de fragmentos",
                options=[
                    "Automática",
                    "Personalizada"
                ],
                index=0,
                help=(
                    "Automática: la cantidad de fragmentos se calcula según "
                    "el tamaño del documento y el tamaño máximo configurado.\n\n"
                    "Personalizada: se intenta aproximar la cantidad indicada, "
                    "pero el tamaño máximo por fragmento siempre tiene prioridad."
                )
            )
        
            if chunk_mode == "Personalizada":
        
                target_chunks = st.number_input(
                    "Cantidad objetivo de fragmentos",
                    min_value=1,
                    max_value=100,
                    value=10,
                    step=1,
                    help=(
                        "Cantidad aproximada de fragmentos que se desea generar. "
                        "El sistema no superará el tamaño máximo configurado."
                    )
                )
        
            else:
        
                target_chunks = None
        st.info("💡 **Tip**: Para papers a 2 columnas, el sistema reordena automáticamente la lectura de la columna izquierda antes de la derecha.")

    # -------------------------------------------------------------------------
    # ÁREA PRINCIPAL: CARGA Y PROCESAMIENTO
    # -------------------------------------------------------------------------

    # Subida de archivos en lote
    uploaded_files = st.file_uploader(
        "Sube uno o varios documentos PDF académicos",
        type=["pdf"],
        accept_multiple_files=True,
        help="Puedes seleccionar múltiples archivos PDF manteniendo presionada la tecla Ctrl/Cmd."
    )

    if uploaded_files:
        st.write(f"📁 **Archivos seleccionados ({len(uploaded_files)}):**")
        cols = st.columns(min(len(uploaded_files), 4))
        for i, file in enumerate(uploaded_files):
            col_idx = i % len(cols)
            cols[col_idx].caption(f"• {file.name} ({round(file.size / 1024, 1)} KB)")

        st.write("")
        start_button = st.button("🚀 Iniciar Traducción de Documentos", type="primary", use_container_width=True)

        if start_button:
            if not api_key:
                st.error("⚠️ Por favor ingresa tu API Key de Gemini en la barra lateral para continuar.")
                return

            # Contenedores para seguimiento en vivo
            overall_progress_bar = st.progress(0.0)
            status_text = st.empty()
            live_container = st.container()

            total_files = len(uploaded_files)
            st.session_state.translated_docs = {}

            # Instanciar el agente de LangChain si se seleccionó ese modo
            agent_executor = None
            if LANGCHAIN_AVAILABLE and "Agente" in translation_mode:
                with st.spinner("Inicializando Agente LangChain con herramientas de traducción..."):
                    try:
                        agent_executor = create_translation_agent_executor(
                            api_key=api_key,
                            target_language=selected_language,
                            model_name=model_choice
                        )
                    except Exception as e:
                        st.error(f"Error al inicializar el agente LangChain: {str(e)}")
                        return

            genai_client = genai.Client(api_key=api_key)
            
            # Iterar por cada documento
            for doc_idx, uploaded_file in enumerate(uploaded_files):
                file_name = uploaded_file.name
                file_bytes = uploaded_file.read()

                status_text.markdown(f"**Procesando documento {doc_idx + 1}/{total_files}:** `{file_name}`...")

                # 1. Extracción de texto con manejo de 2 columnas y detección de escaneados
                try:
                    raw_text, detected_title, is_scanned = extract_pdf_content(file_bytes, file_name)
                except Exception as e:
                    st.error(f"❌ Error al leer el PDF '{file_name}': {str(e)}")
                    continue

                # Manejo de error para PDFs escaneados
                if is_scanned or not raw_text.strip():
                    st.warning(
                        f"⚠️ **Atención:** El archivo `{file_name}` no contiene una capa de texto digital extraíble "
                        "(parece ser un PDF escaneado o una imagen). Por favor proporciona un PDF digitalizado con texto seleccionable."
                    )
                    continue

                # 2. Segmentación en fragmentos inteligentes
                chunks = segment_text_into_chunks(
                    raw_text,
                    max_chars=chunk_size,
                    target_chunks=target_chunks
                )
                total_chunks = len(chunks)

                if total_chunks == 0:
                    st.warning(f"No se pudieron generar fragmentos de texto para `{file_name}`.")
                    continue

                # 3. Traducción concurrente con visualización en vivo

                # Crear el cliente de Gemini una sola vez.
                gemini_client = genai.Client(api_key=api_key)
                
                translated_chunks: List[str] = [None] * total_chunks
                
                
                def update_translation_progress(
                    completed: int,
                    total: int,
                    completed_chunk_index: int,
                    completed_translation: str
                ):
                    """
                    Actualiza el progreso y la vista en vivo cuando termina
                    cualquiera de los chunks concurrentes.
                    """
                
                    # Progreso del documento actual.
                    document_progress = completed / total
                
                    current_overall = (
                        doc_idx + document_progress
                    ) / total_files
                
                    overall_progress_bar.progress(
                        min(current_overall, 1.0)
                    )
                
                    status_text.markdown(
                        f"📄 **Doc {doc_idx + 1}/{total_files}** (`{file_name}`) — "
                        f"Traducidos **{completed}/{total} fragmentos** "
                        f"al {selected_language}..."
                    )
                
                    # Mostrar el último fragmento que haya terminado.
                    original_chunk = chunks[completed_chunk_index]
                
                    live_container.empty()
                
                    with live_container.container():
                
                        st.markdown(
                            f"##### 🔍 Vista en Vivo — `{file_name}` "
                            f"(Fragmento {completed_chunk_index + 1}/{total})"
                        )
                
                        c1, c2 = st.columns(2)
                
                        with c1:
                            st.caption(
                                "📝 Texto Original "
                                "(Extracción 2 columnas)"
                            )
                
                            st.text_area(
                                "Original",
                                value=(
                                    original_chunk[:500]
                                    + ("..." if len(original_chunk) > 500 else "")
                                ),
                                height=110,
                                key=f"live_orig_{doc_idx}_{completed_chunk_index}",
                                disabled=True,
                                label_visibility="collapsed"
                            )
                
                        with c2:
                            st.caption(
                                f"✨ Traducción ({selected_language})"
                            )
                
                            st.text_area(
                                "Traducido",
                                value=(
                                    completed_translation[:500]
                                    + (
                                        "..."
                                        if len(completed_translation) > 500
                                        else ""
                                    )
                                ),
                                height=110,
                                key=f"live_trans_{doc_idx}_{completed_chunk_index}",
                                disabled=True,
                                label_visibility="collapsed"
                            )
                
                
                # =========================================================================
                # Ejecutar traducción según el modo seleccionado
                # =========================================================================
                
                if "Agente" in translation_mode:
                
                    translated_chunks = translate_chunks_agent_concurrent(
                        chunks=chunks,
                        target_language=selected_language,
                        agent_executor=agent_executor,
                        max_workers=MAX_CONCURRENT_REQUESTS,
                        progress_callback=update_translation_progress
                    )
                
                else:
                
                    if "Batch" in translation_mode:

                        translated_chunks = translate_chunks_batch(
                            chunks=chunks,
                            target_language=selected_language,
                            client=gemini_client,
                            model_name=model_choice,
                            progress_callback=update_translation_progress
                        )
                    
                    else:
                    
                        translated_chunks = translate_chunks_concurrent(
                            chunks=chunks,
                            target_language=selected_language,
                            client=gemini_client,
                            model_name=model_choice,
                            max_workers=MAX_CONCURRENT_REQUESTS,
                            progress_callback=update_translation_progress
                        )

                # Unir el texto traducido completo para este documento
                full_translated_doc = "\n\n".join(translated_chunks)
                
                # Usar el título detectado o el nombre del archivo
                doc_title = detected_title if detected_title else file_name.replace(".pdf", "").replace("_", " ")

                # -------------------------------------------------------------
                # Generar .docx
                # -------------------------------------------------------------
                docx_file_io = generate_docx(
                    title=doc_title,
                    text_content=full_translated_doc,
                    source_filename=file_name
                )
                
                # -------------------------------------------------------------
                # Generar .txt
                # -------------------------------------------------------------
                txt_bytes = generate_txt(
                    text_content=full_translated_doc
                )
                
                # -------------------------------------------------------------
                # Guardar resultados en session_state
                # -------------------------------------------------------------
                st.session_state.translated_docs[file_name] = {
                    "text": full_translated_doc,
                    "title": doc_title,
                    "docx_bytes": docx_file_io.getvalue(),
                    "txt_bytes": txt_bytes
                }

            overall_progress_bar.progress(1.0)
            status_text.success("🎉 ¡Traducción de todos los documentos completada con éxito!")
            st.session_state.processing_complete = True

    # -------------------------------------------------------------------------
    # SECCIÓN DE REVISIÓN EDITABLE Y DESCARGAS
    # -------------------------------------------------------------------------
    if st.session_state.translated_docs:
        st.divider()
        st.header("✏️ Revisión, Edición y Descargas")
        st.markdown(
            "Puedes revisar y corregir el texto traducido en los cuadros editables a continuación. "
            "Al hacer clic en **Actualizar DOCX**, los cambios se aplicarán al archivo final."
        )

        zip_files_dict: Dict[str, bytes] = {}

        for filename, doc_data in st.session_state.translated_docs.items():
            base_name = filename.rsplit('.', 1)[0]
            docx_filename = f"{base_name}_traducido_{selected_language.lower()}.docx"

            with st.expander(f"📄 {filename} → {docx_filename}", expanded=True):
                # Título editable
                edit_title = st.text_input(
                    f"Título del Documento ({filename})",
                    value=doc_data["title"],
                    key=f"title_{filename}"
                )

                # Área de texto editable para revisión humana
                edited_text = st.text_area(
                    f"Contenido traducido editable ({filename})",
                    value=doc_data["text"],
                    height=280,
                    key=f"textarea_{filename}"
                )

                # Si el usuario editó el texto o título, actualizar el docx en memoria
                if (edited_text != doc_data["text"]) or (edit_title != doc_data["title"]):

                    # Regenerar DOCX
                    updated_docx_io = generate_docx(
                        title=edit_title,
                        text_content=edited_text,
                        source_filename=filename
                    )
                
                    # Regenerar TXT
                    updated_txt_bytes = generate_txt(
                        text_content=edited_text
                    )
                
                    # Actualizar memoria
                    doc_data["text"] = edited_text
                    doc_data["title"] = edit_title
                    doc_data["docx_bytes"] = updated_docx_io.getvalue()
                    doc_data["txt_bytes"] = updated_txt_bytes

                # Nombre del TXT
                base_name = filename.rsplit('.', 1)[0]
                
                txt_filename = (
                    f"{base_name}_traducido_{selected_language.lower()}.txt"
                )
                
                # Agregar ambos archivos al ZIP
                zip_files_dict[docx_filename] = doc_data["docx_bytes"]
                zip_files_dict[txt_filename] = doc_data["txt_bytes"]

                # -------------------------------------------------------------
                # Botones de descarga
                # -------------------------------------------------------------
                
                download_col1, download_col2 = st.columns(2)
                
                with download_col1:
                
                    st.download_button(
                        label=f"📥 Descargar DOCX",
                        data=doc_data["docx_bytes"],
                        file_name=docx_filename,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key=f"download_docx_{filename}"
                    )
                    


                
                with download_col2:
                
                    base_name = filename.rsplit('.', 1)[0]
                
                    txt_filename = (
                        f"{base_name}_traducido_{selected_language.lower()}.txt"
                    )
                
                    st.download_button(
                        label=f"📄 Descargar TXT",
                        data=doc_data["txt_bytes"],
                        file_name=txt_filename,
                        mime="text/plain",
                        key=f"download_txt_{filename}"
                    )

        st.divider()

        # Botón para descargar todos los archivos en un ZIP
        if zip_files_dict:
        
            zip_bytes = generate_zip_package(
                zip_files_dict
            )
        
            st.download_button(
                label=(
                    f"📦 Descargar TODOS los documentos "
                    f"(.ZIP) [{len(zip_files_dict)} archivos]"
                ),
                data=zip_bytes,
                file_name=(
                    f"papers_traducidos_"
                    f"{selected_language.lower()}.zip"
                ),
                mime="application/zip",
                type="primary",
                use_container_width=True
            )


if __name__ == "__main__":
    main()
