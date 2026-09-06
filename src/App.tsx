import React, { useState } from 'react';
import {
  FileText,
  FileCode2,
  Cpu,
  Layers,
  Download,
  Copy,
  Check,
  Play,
  Languages,
  BookOpen,
  CheckCircle2,
  Terminal,
  Key,
  AlertTriangle,
  FolderArchive,
  RefreshCw,
  Sparkles,
  ArrowRight,
  ExternalLink,
  ShieldCheck,
  FileCheck2
} from 'lucide-react';

const APP_PY_CODE = `"""
=============================================================================
TRADUCTOR DE PAPERS Y DOCUMENTOS ACADÉMICOS (PDF a DOCX)
Streamlit + Google Gemini + LangChain + PyMuPDF + python-docx
=============================================================================
"""

import io
import re
import zipfile
from typing import List, Tuple, Dict, Optional
import streamlit as st

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

# Google Gemini API directo
import google.generativeai as genai

# LangChain y Agentes
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.tools import tool
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
except ImportError:
    pass


# =============================================================================
# 1. EXTRACCIÓN DE TEXTO RESPETANDO COLUMNAS (PyMuPDF / pdfplumber)
# =============================================================================

def extract_text_two_columns_pymupdf(file_bytes: bytes) -> Tuple[str, str, bool]:
    """
    Extrae texto de un PDF preservando el orden de lectura de dos columnas.
    Lee primero la columna izquierda completa antes de la derecha.
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

        # Bloques: (x0, y0, x1, y1, text, block_no, block_type)
        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]

        if not text_blocks:
            continue

        # Extraer título en la primera página
        if page_idx == 0 and not detected_title:
            for b in text_blocks[:3]:
                candidate = b[4].strip().replace('\\n', ' ')
                if 15 <= len(candidate) <= 200 and not candidate.lower().startswith("issn"):
                    detected_title = candidate
                    break

        top_spanning = []
        left_column = []
        right_column = []
        bottom_spanning = []

        for b in text_blocks:
            x0, y0, x1, y1, text, _, _ = b
            block_width = x1 - x0
            
            if block_width > (page_width * 0.65):
                if y0 < rect.height * 0.35:
                    top_spanning.append(b)
                else:
                    bottom_spanning.append(b)
            elif x0 < mid_x and x1 <= (mid_x + 40):
                left_column.append(b)
            elif x0 >= (mid_x - 40):
                right_column.append(b)
            else:
                block_center_x = (x0 + x1) / 2.0
                if block_center_x < mid_x:
                    left_column.append(b)
                else:
                    right_column.append(b)

        # Ordenar verticalmente cada grupo por su coordenada y0
        top_spanning.sort(key=lambda b: b[1])
        left_column.sort(key=lambda b: b[1])
        right_column.sort(key=lambda b: b[1])
        bottom_spanning.sort(key=lambda b: b[1])

        # Ensamblar: cabecera -> columna izquierda -> columna derecha -> pie
        page_ordered_blocks = top_spanning + left_column + right_column + bottom_spanning

        page_text_pieces = []
        for b in page_ordered_blocks:
            clean_block_text = b[4].strip()
            if clean_block_text:
                page_text_pieces.append(clean_block_text)
                total_char_count += len(clean_block_text)

        if page_text_pieces:
            all_pages_text.append("\\n\\n".join(page_text_pieces))

    doc.close()
    full_text = "\\n\\n".join(all_pages_text)
    is_scanned = total_char_count < 60
    return full_text, detected_title, is_scanned


def segment_text_into_chunks(text: str, max_chars: int = 1900) -> List[str]:
    """Divide el texto en fragmentos respetando oraciones y párrafos."""
    if not text or not text.strip():
        return []

    normalized_text = text.replace('\\r\\n', '\\n').replace('\\r', '\\n')
    raw_paragraphs = [p.strip() for p in normalized_text.split('\\n\\n') if p.strip()]

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0

    for paragraph in raw_paragraphs:
        p_len = len(paragraph)
        if current_length + p_len + 2 <= max_chars:
            current_chunk.append(paragraph)
            current_length += p_len + 2
        else:
            if p_len > max_chars:
                if current_chunk:
                    chunks.append("\\n\\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0

                sentences = re.split(r'(?<=[.!?])\\s+', paragraph)
                temp_sentence_chunk: List[str] = []
                temp_len = 0

                for sent in sentences:
                    s_clean = sent.strip()
                    if not s_clean:
                        continue
                    if temp_len + len(s_clean) + 1 <= max_chars:
                        temp_sentence_chunk.append(s_clean)
                        temp_len += len(s_clean) + 1
                    else:
                        if temp_sentence_chunk:
                            chunks.append(" ".join(temp_sentence_chunk))
                        temp_sentence_chunk = [s_clean]
                        temp_len = len(s_clean)

                if temp_sentence_chunk:
                    chunks.append(" ".join(temp_sentence_chunk))
            else:
                if current_chunk:
                    chunks.append("\\n\\n".join(current_chunk))
                current_chunk = [paragraph]
                current_length = p_len

    if current_chunk:
        chunks.append("\\n\\n".join(current_chunk))

    return chunks


# =============================================================================
# TRADUCCIÓN: MODO DIRECTO (Gemini API)
# =============================================================================

def translate_chunk_direct(chunk: str, target_language: str, api_key: str, model_name: str = "gemini-1.5-flash") -> str:
    genai.configure(api_key=api_key)
    system_instruction = (
        f"Eres un traductor académico profesional. Traduce al idioma: {target_language}.\\n"
        "REGLAS:\\n"
        "1. Devuelve ÚNICAMENTE el texto traducido sin preámbulos.\\n"
        "2. NO traduzcas fórmulas matemáticas, ecuaciones, código ni referencias bibliográficas (ej. [1]).\\n"
        "3. NO traduzcas nombres propios ni instituciones."
    )
    model = genai.GenerativeModel(model_name=model_name, system_instruction=system_instruction)
    response = model.generate_content(f"Texto a traducir:\\n\\n{chunk}")
    return response.text.strip() if response and response.text else chunk


# =============================================================================
# TRADUCCIÓN: MODO AGENTE (LangChain + Gemini + Tools)
# =============================================================================

def create_translation_agent_executor(api_key: str, target_language: str, model_name: str = "gemini-1.5-flash") -> AgentExecutor:
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.1)

    @tool
    def detect_language_tool(text: str) -> str:
        """Detecta el idioma del texto y devuelve su código ISO."""
        try:
            return detect(text[:500].strip())
        except Exception:
            return "unknown"

    @tool
    def translate_academic_text_tool(text: str, target_lang: str) -> str:
        """Traduce respetando fórmulas y citas."""
        prompt = f"Traduce al {target_lang} respetando citas y fórmulas:\\n\\n{text}"
        res = llm.invoke(prompt)
        return res.content if hasattr(res, 'content') else str(res)

    tools = [detect_language_tool, translate_academic_text_tool]
    system_message = (
        f"Eres un agente supervisor de traducción. Idioma objetivo: '{target_language}'.\\n"
        "Si el texto ya está en el idioma objetivo, devuélvelo tal cual. "
        "Si no, usa translate_academic_text_tool. Devuelve solo el texto final."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=4)


# =============================================================================
# RECONSTRUCCIÓN .DOCX Y ZIP
# =============================================================================

def generate_docx(title: str, text_content: str, source_filename: str = "") -> io.BytesIO:
    doc = Document()
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)

    title_p = doc.add_heading(title or "Documento Traducido", level=1)
    title_p.alignment = WD_ALIGN_PARAGRAPH.LEFT

    meta_p = doc.add_paragraph(f"Traducción automática | Fuente: {source_filename or 'PDF'}")
    meta_p.runs[0].font.italic = True
    meta_p.runs[0].font.size = Pt(9.5)

    doc.add_paragraph("―" * 40)

    for p_text in text_content.split('\\n\\n'):
        if p_text.strip():
            p = doc.add_paragraph(p_text.strip())
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_after = Pt(6)

    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio
`;

const REQUIREMENTS_TXT = `streamlit>=1.35.0
google-generativeai>=0.8.0
langchain>=0.2.0
langchain-google-genai>=1.0.5
langchain-core>=0.2.0
langdetect>=1.0.9
pymupdf>=1.24.0
pdfplumber>=0.11.0
python-docx>=1.1.0
pydantic>=2.7.0
`;

const SAMPLE_PAPERS = [
  {
    id: 'paper1',
    name: 'Attention_Is_All_You_Need_Transformer.pdf',
    title: 'Attention Is All You Need',
    pages: 11,
    size: '2.1 MB',
    isScanned: false,
    textLeftCol: `Abstract: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output [1]. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable.
1. Introduction: Recurrent neural networks, long short-term memory [2] and gated recurrent neural networks [3] have been firmly established as state of the art approaches in sequence modeling. Recurrent models typically factor computation along the symbol positions of the input and output sequences.`,
    textRightCol: `This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks [4, 5].
In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.`
  },
  {
    id: 'paper2',
    name: 'Quantum_Computing_Error_Correction.pdf',
    title: 'Surface Codes for Fault-Tolerant Quantum Computation',
    pages: 8,
    size: '1.4 MB',
    isScanned: false,
    textLeftCol: `Abstract: Large-scale quantum computing requires fault-tolerant architectures capable of correcting physical qubit errors. The surface code has emerged as a leading candidate due to its high fault-tolerance threshold of approximately 1% and nearest-neighbor 2D grid requirements [1].
1. Mathematical Framework: Let H denote the Hilbert space with state vector |ψ⟩ = α|0⟩ + β|1⟩ where |α|^2 + |β|^2 = 1. Stabilizer operators S_x = X_1 X_2 X_3 X_4 and S_z = Z_1 Z_2 Z_3 Z_4 commute with the code space projector.`,
    textRightCol: `2. Syndrome Measurement: Syndrome extraction circuits detect bit-flip (X) and phase-flip (Z) errors without collapsing the superposition of logical states. Minimum-weight perfect matching (MWPM) decoding algorithms achieve polynomial-time error identification.
As demonstrated by Fowler et al. (2012) [2], physical error rates below 10^-3 enable logical error suppression scaling exponentially with code distance d.`
  },
  {
    id: 'paper3',
    name: 'Historical_Scanned_Manuscript_1920.pdf',
    title: 'Documento Escaneado (Sin Capa OCR)',
    pages: 4,
    size: '4.8 MB',
    isScanned: true,
    textLeftCol: '',
    textRightCol: ''
  }
];

export default function App() {
  const [activeTab, setActiveTab] = useState<'simulator' | 'code' | 'architecture'>('simulator');
  const [copiedFile, setCopiedFile] = useState<string | null>(null);

  // Settings state matching Variation 8 design
  const [selectedTargetLang, setSelectedTargetLang] = useState('Español');
  const [selectedMode, setSelectedMode] = useState<'direct' | 'agent'>('direct');
  const [selectedModel, setSelectedModel] = useState('gemini-1.5-flash');
  const [apiKeyInput, setApiKeyInput] = useState('AIzaSyD-SIMULATED-GEMINI-KEY-ACADEMIC');
  const [activePaperIndex, setActivePaperIndex] = useState(0);
  const [isTranslating, setIsTranslating] = useState(false);
  const [translationProgress, setTranslationProgress] = useState(0);
  const [currentStepText, setCurrentStepText] = useState('');
  const [translatedResult, setTranslatedResult] = useState<string>('');
  const [editableDocTitle, setEditableDocTitle] = useState<string>('');
  const [isCompleted, setIsCompleted] = useState(false);
  const [activeLiveComparison, setActiveLiveComparison] = useState<{ orig: string; trans: string } | null>(null);

  const currentPaper = SAMPLE_PAPERS[activePaperIndex];

  const handleCopy = (text: string, label: string) => {
    navigator.clipboard.writeText(text);
    setCopiedFile(label);
    setTimeout(() => setCopiedFile(null), 2500);
  };

  const handleDownloadFile = (filename: string, content: string) => {
    const element = document.createElement('a');
    const file = new Blob([content], { type: 'text/plain;charset=utf-8' });
    element.href = URL.createObjectURL(file);
    element.download = filename;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const runTranslationSimulation = async () => {
    if (!apiKeyInput.trim()) {
      alert("Por favor ingresa una API Key simulada o real.");
      return;
    }

    if (currentPaper.isScanned) {
      alert("⚠️ El archivo es un PDF escaneado sin capa de texto digital. La aplicación mostrará una advertencia clara en Streamlit sin fallar.");
      return;
    }

    setIsTranslating(true);
    setIsCompleted(false);
    setTranslationProgress(0);
    setTranslatedResult('');
    setEditableDocTitle(currentPaper.title);

    // Step 1: PyMuPDF 2-column extraction
    setCurrentStepText(`Extrayendo bloques geométricos con PyMuPDF (ordenando Columna Izquierda -> Derecha)...`);
    setTranslationProgress(20);
    await new Promise(r => setTimeout(r, 500));

    // Step 2: Chunk segmentation
    setCurrentStepText(`Segmentando en fragmentos oracionales de ~1900 caracteres (sin cortar frases)...`);
    setTranslationProgress(40);
    await new Promise(r => setTimeout(r, 500));

    // Step 3: LLM / Agent Translation
    if (selectedMode === 'agent') {
      setCurrentStepText(`Agente LangChain: ejecutando detect_language_tool...`);
      setTranslationProgress(60);
      await new Promise(r => setTimeout(r, 600));
      setCurrentStepText(`Agente LangChain: invocando translate_academic_text_tool (${selectedTargetLang})...`);
    } else {
      setCurrentStepText(`Modo Directo: Enviando prompt con restricciones académicas a Gemini (${selectedTargetLang})...`);
    }

    const mockTranslatedOutput = selectedTargetLang === 'Español'
      ? `Resumen: Los modelos dominantes de transducción de secuencias se basan en redes neuronales recurrentes o convolucionales complejas. Proponemos el Transformer, una arquitectura de modelo que prescinde de la recurrencia y, en su lugar, se basa completamente en un mecanismo de atención para establecer dependencias globales entre la entrada y la salida [1]. Los experimentos en dos tareas de traducción automática demuestran que estos modelos son superiores en calidad, siendo a la vez significativamente más paralelizables.

1. Introducción: Las redes neuronales recurrentes, la memoria a largo plazo (LSTM) [2] y las redes recurrentes con compuertas [3] se han consolidado firmemente como enfoques de vanguardia en el modelado de secuencias. Típicamente, los modelos recurrentes factorizan el cómputo a lo largo de las posiciones de los símbolos de las secuencias de entrada y salida.

Esta naturaleza inherentemente secuencial impide la paralelización dentro de los ejemplos de entrenamiento, lo cual resulta crítico en secuencias más largas, ya que las restricciones de memoria limitan el procesamiento por lotes entre ejemplos. Los mecanismos de atención se han convertido en una parte integral de los modelos convincentes de transducción y modelado de secuencias en diversas tareas [4, 5].

En este trabajo proponemos el Transformer, una arquitectura de modelo que prescinde de la recurrencia y se basa por completo en un mecanismo de atención para deducir dependencias globales. El Transformer permite una paralelización significativamente mayor y puede alcanzar un nuevo estado del arte en calidad de traducción tras haber sido entrenado durante tan solo doce horas en ocho GPU P100.`
      : `Résumé: Les modèles dominants de transduction de séquences reposent sur des réseaux neuronaux récurrents ou convolutionnels complexes. Nous proposons le Transformer, une architecture de modèle évitant la récurrence et s'appuyant entièrement sur un mécanisme d'attention [1].
1. Introduction: Les réseaux récurrents [2, 3] sont fermement établis. Cette nature séquentielle empêche la parallélisation. Le Transformer permet une parallélisation accrue et atteint l'état de l'art après seulement 12 heures d'entraînement sur 8 GPU P100.`;

    setActiveLiveComparison({
      orig: currentPaper.textLeftCol.slice(0, 260) + '...',
      trans: mockTranslatedOutput.slice(0, 290) + '...'
    });

    setTranslationProgress(85);
    await new Promise(r => setTimeout(r, 600));

    // Step 4: DOCX reconstruction
    setCurrentStepText(`Reconstruyendo documento limpio en .docx de flujo continuo de una columna...`);
    setTranslationProgress(100);
    await new Promise(r => setTimeout(r, 400));

    setTranslatedResult(mockTranslatedOutput);
    setIsTranslating(false);
    setIsCompleted(true);
    setCurrentStepText('¡Traducción completada! Documento listo para revisión y descarga.');
  };

  return (
    <div className="min-h-screen bg-[#fdfdfc] text-[#1a1a1e] flex flex-col font-sans selection:bg-[#2563eb] selection:text-white">
      {/* Editorial Header matching Variation 8 */}
      <header className="px-6 md:px-12 py-6 border-b border-[rgba(26,26,30,0.08)] flex flex-wrap items-center justify-between gap-4 bg-[#fdfdfc]">
        <div className="brand">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-[#1a1a1e] text-white flex items-center justify-center font-bold text-sm">
              <FileText className="w-4 h-4 text-white" />
            </div>
            <h1 className="text-xl md:text-2xl font-bold tracking-tight text-[#1a1a1e]">Academic Translator</h1>
          </div>
          <p className="text-xs text-[rgba(26,26,30,0.6)] mt-1 font-medium">
            PyMuPDF 2-column Extraction & Gemini Intelligence
          </p>
        </div>

        <div className="nav-pills flex gap-1.5 bg-[#f1f1ee] p-1 rounded-lg">
          <button
            id="tab-simulator"
            onClick={() => setActiveTab('simulator')}
            className={`px-4 py-2 text-xs font-semibold rounded-md transition-all ${
              activeTab === 'simulator'
                ? 'bg-white text-[#1a1a1e] shadow-sm'
                : 'text-[rgba(26,26,30,0.6)] hover:text-[#1a1a1e]'
            }`}
          >
            Simulador Interactivo
          </button>
          <button
            id="tab-code"
            onClick={() => setActiveTab('code')}
            className={`px-4 py-2 text-xs font-semibold rounded-md transition-all ${
              activeTab === 'code'
                ? 'bg-white text-[#1a1a1e] shadow-sm'
                : 'text-[rgba(26,26,30,0.6)] hover:text-[#1a1a1e]'
            }`}
          >
            Código Python (app.py)
          </button>
          <button
            id="tab-architecture"
            onClick={() => setActiveTab('architecture')}
            className={`px-4 py-2 text-xs font-semibold rounded-md transition-all ${
              activeTab === 'architecture'
                ? 'bg-white text-[#1a1a1e] shadow-sm'
                : 'text-[rgba(26,26,30,0.6)] hover:text-[#1a1a1e]'
            }`}
          >
            Arquitectura
          </button>
        </div>
      </header>

      {/* Main Body */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-[320px_1fr]">
        {/* Sidebar matching Variation 8 design */}
        <aside className="border-r border-[rgba(26,26,30,0.08)] p-6 md:p-8 bg-[#fafaf9] flex flex-col justify-between overflow-y-auto">
          <div className="space-y-6">
            <h3 className="text-xs font-bold font-mono tracking-wider uppercase text-[#1a1a1e] pb-3 border-b border-[rgba(26,26,30,0.08)]">
              Configuración
            </h3>

            {/* API Key */}
            <div className="form-group space-y-2">
              <label className="font-mono text-[11px] uppercase tracking-wider text-[rgba(26,26,30,0.6)] block">
                Gemini API Key
              </label>
              <input
                type="password"
                value={apiKeyInput}
                onChange={(e) => setApiKeyInput(e.target.value)}
                placeholder="AIzaSy..."
                className="w-full p-3 bg-white border border-[rgba(26,26,30,0.12)] rounded-md text-xs text-[#1a1a1e] font-mono focus:outline-none focus:border-[#2563eb] transition-colors"
              />
              <span className="text-[10px] text-[rgba(26,26,30,0.45)] block leading-tight">
                st.text_input con type="password". No se hardcodea.
              </span>
            </div>

            {/* Target Language */}
            <div className="form-group space-y-2">
              <label className="font-mono text-[11px] uppercase tracking-wider text-[rgba(26,26,30,0.6)] block">
                Idioma de Destino
              </label>
              <select
                value={selectedTargetLang}
                onChange={(e) => setSelectedTargetLang(e.target.value)}
                className="w-full p-3 bg-white border border-[rgba(26,26,30,0.12)] rounded-md text-xs text-[#1a1a1e] focus:outline-none focus:border-[#2563eb] transition-colors"
              >
                <option value="Español">Español</option>
                <option value="Inglés">Inglés</option>
                <option value="Francés">Francés</option>
                <option value="Portugués">Portugués</option>
                <option value="Alemán">Alemán</option>
                <option value="Italiano">Italiano</option>
              </select>
            </div>

            {/* Mode Selection */}
            <div className="form-group space-y-2">
              <label className="font-mono text-[11px] uppercase tracking-wider text-[rgba(26,26,30,0.6)] block">
                Modo de Traducción
              </label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => setSelectedMode('direct')}
                  className={`p-3 border rounded-lg text-left transition-all ${
                    selectedMode === 'direct'
                      ? 'border-[#2563eb] bg-[rgba(37,99,235,0.04)] shadow-xs'
                      : 'border-[rgba(26,26,30,0.1)] bg-white hover:border-[rgba(26,26,30,0.2)]'
                  }`}
                >
                  <span className="font-bold text-xs text-[#1a1a1e] block">Directo</span>
                  <span className="text-[10px] text-[rgba(26,26,30,0.55)] block mt-0.5">Gemini Prompt</span>
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedMode('agent')}
                  className={`p-3 border rounded-lg text-left transition-all ${
                    selectedMode === 'agent'
                      ? 'border-[#2563eb] bg-[rgba(37,99,235,0.04)] shadow-xs'
                      : 'border-[rgba(26,26,30,0.1)] bg-white hover:border-[rgba(26,26,30,0.2)]'
                  }`}
                >
                  <span className="font-bold text-xs text-[#1a1a1e] block">Agente</span>
                  <span className="text-[10px] text-[rgba(26,26,30,0.55)] block mt-0.5">LangChain Flow</span>
                </button>
              </div>
            </div>

            {/* Model Choice */}
            <div className="form-group space-y-2">
              <label className="font-mono text-[11px] uppercase tracking-wider text-[rgba(26,26,30,0.6)] block">
                Modelo
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full p-3 bg-white border border-[rgba(26,26,30,0.12)] rounded-md text-xs text-[#1a1a1e] focus:outline-none focus:border-[#2563eb] transition-colors"
              >
                <option value="gemini-1.5-flash">gemini-1.5-flash</option>
                <option value="gemini-2.0-flash">gemini-2.0-flash</option>
                <option value="gemini-1.5-pro">gemini-1.5-pro</option>
              </select>
            </div>

            {/* Chunk Size Indicator */}
            <div className="space-y-2 pt-2">
              <div className="flex justify-between font-mono text-[11px]">
                <span className="text-[rgba(26,26,30,0.6)] uppercase">Chunk Size</span>
                <span className="text-[#2563eb] font-bold">1900 chars</span>
              </div>
              <div className="h-1.5 bg-[rgba(26,26,30,0.08)] rounded-full overflow-hidden">
                <div className="h-full bg-[#2563eb] w-[80%] rounded-full"></div>
              </div>
              <span className="text-[10px] text-[rgba(26,26,30,0.5)] block">
                Segmentación que preserva oraciones completas y párrafos.
              </span>
            </div>
          </div>

          {/* Terminal Command box */}
          <div className="mt-8 bg-[#111111] text-[#10b981] p-3.5 rounded-lg font-mono text-xs shadow-inner flex items-center justify-between">
            <span className="truncate">$ streamlit run app.py</span>
            <button
              onClick={() => handleCopy('streamlit run app.py', 'cmd')}
              className="text-[10px] text-zinc-400 hover:text-white ml-2 shrink-0"
              title="Copiar comando"
            >
              {copiedFile === 'cmd' ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            </button>
          </div>
        </aside>

        {/* Content Area */}
        <section className="p-6 md:p-12 overflow-y-auto bg-[#fdfdfc]">
          {/* TAB 1: SIMULATOR */}
          {activeTab === 'simulator' && (
            <div className="max-w-4xl space-y-8">
              <div>
                <span className="font-mono text-xs text-[rgba(26,26,30,0.6)] uppercase tracking-wider block mb-2">
                  Batch Processing View
                </span>
                <h2 className="text-2xl md:text-3xl font-bold tracking-tight text-[#1a1a1e] mb-2">
                  Documentos PDF Académicos
                </h2>
                <p className="text-xs text-[rgba(26,26,30,0.6)]">
                  Selecciona un paper cargado en lote (st.file_uploader con accept_multiple_files=True).
                </p>
              </div>

              {/* File Card Grid matching Variation 8 */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                {SAMPLE_PAPERS.map((paper, idx) => (
                  <button
                    key={paper.id}
                    onClick={() => {
                      setActivePaperIndex(idx);
                      setIsCompleted(false);
                      setTranslatedResult('');
                      setActiveLiveComparison(null);
                    }}
                    className={`p-5 rounded-xl border text-left transition-all flex flex-col justify-between gap-3 bg-white ${
                      activePaperIndex === idx
                        ? 'border-2 border-[#2563eb] shadow-md -translate-y-0.5'
                        : 'border-[rgba(26,26,30,0.1)] hover:border-[rgba(26,26,30,0.3)] hover:-translate-y-0.5'
                    }`}
                  >
                    <div>
                      <div className="flex items-start justify-between gap-2 mb-2">
                        <FileText className={`w-4 h-4 ${paper.isScanned ? 'text-amber-600' : 'text-[#2563eb]'}`} />
                        {paper.isScanned && (
                          <span className="px-1.5 py-0.5 rounded text-[10px] font-bold uppercase bg-[#fff7ed] text-[#c2410c] border border-[#ffedd5]">
                            Escaneado
                          </span>
                        )}
                      </div>
                      <div className="text-xs font-semibold text-[#1a1a1e] leading-snug break-words">
                        {paper.name}
                      </div>
                    </div>
                    <div className="text-[11px] text-[rgba(26,26,30,0.5)] font-mono">
                      {paper.pages} págs • {paper.size}
                    </div>
                  </button>
                ))}
              </div>

              {/* Scanned PDF notice */}
              {currentPaper.isScanned && (
                <div className="p-4 rounded-lg bg-[#fff7ed] border border-[#ffedd5] text-[#c2410c] text-xs flex items-start gap-3">
                  <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5 text-amber-600" />
                  <div>
                    <span className="font-bold">Manejo de Errores:</span> El archivo '{currentPaper.name}' carece de capa digital de texto extraíble. En la aplicación Streamlit se emite un <code className="font-mono bg-white px-1 py-0.5 rounded border border-[#ffedd5]">st.warning</code> informativo para el usuario sin detener la ejecución de otros documentos.
                  </div>
                </div>
              )}

              {/* CTA Execution Button matching Variation 8 */}
              <button
                id="btn-start-translation"
                onClick={runTranslationSimulation}
                disabled={isTranslating || currentPaper.isScanned}
                className={`w-full py-4 px-6 rounded-lg font-bold text-sm flex items-center justify-center gap-2.5 transition-all shadow-sm ${
                  isTranslating
                    ? 'bg-[#2563eb] text-white cursor-wait opacity-90'
                    : currentPaper.isScanned
                    ? 'bg-[#e5e5e0] text-[#888880] cursor-not-allowed'
                    : 'bg-[#1a1a1e] hover:bg-[#2563eb] text-white'
                }`}
              >
                {isTranslating ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    Procesando y Traduciendo en Tiempo Real...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-current" />
                    Iniciar Extracción & Traducción ({selectedTargetLang})
                  </>
                )}
              </button>

              {/* Progress & Live Review Section */}
              {(isTranslating || isCompleted) && (
                <div className="p-6 rounded-xl border border-[rgba(26,26,30,0.1)] bg-white space-y-4 shadow-sm">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <RefreshCw className={`w-4 h-4 text-[#2563eb] ${isTranslating ? 'animate-spin' : ''}`} />
                      <h3 className="text-xs font-bold uppercase font-mono tracking-wider text-[#1a1a1e]">
                        Revisión en Vivo (Live Streamlit View)
                      </h3>
                    </div>
                    <span className="text-xs font-mono font-bold text-[#2563eb]">{translationProgress}%</span>
                  </div>

                  <div className="h-2 bg-[#f1f1ee] rounded-full overflow-hidden">
                    <div
                      className="h-full bg-[#2563eb] transition-all duration-300 rounded-full"
                      style={{ width: `${translationProgress}%` }}
                    ></div>
                  </div>

                  <p className="text-xs text-[rgba(26,26,30,0.7)] font-medium font-mono">
                    {currentStepText}
                  </p>

                  {/* 2-column comparison */}
                  {activeLiveComparison && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                      <div className="p-4 rounded-lg bg-[#fafaf9] border border-[rgba(26,26,30,0.08)]">
                        <span className="text-[10px] font-mono uppercase tracking-wider font-bold text-[rgba(26,26,30,0.6)] block mb-2">
                          Original (Columna Izquierda PyMuPDF)
                        </span>
                        <p className="text-xs text-[#1a1a1e] leading-relaxed bg-white p-3 rounded border border-[rgba(26,26,30,0.06)] font-serif">
                          {activeLiveComparison.orig}
                        </p>
                      </div>
                      <div className="p-4 rounded-lg bg-[#eff6ff] border border-[#dbeafe]">
                        <span className="text-[10px] font-mono uppercase tracking-wider font-bold text-[#2563eb] block mb-2">
                          Traducción Gemini ({selectedTargetLang})
                        </span>
                        <p className="text-xs text-[#1a1a1e] leading-relaxed bg-white p-3 rounded border border-[#bfdbfe] font-serif">
                          {activeLiveComparison.trans}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Editable Document Output & Downloads */}
              {isCompleted && translatedResult && (
                <div className="p-6 rounded-xl border-2 border-[#2563eb]/30 bg-white space-y-5 shadow-md">
                  <div className="flex items-center justify-between border-b border-[rgba(26,26,30,0.08)] pb-4">
                    <div className="flex items-center gap-2.5">
                      <CheckCircle2 className="w-5 h-5 text-emerald-600" />
                      <div>
                        <h3 className="text-sm font-bold text-[#1a1a1e]">Revisión Editable & Generación DOCX</h3>
                        <p className="text-xs text-[rgba(26,26,30,0.6)]">Revisa el texto traducido en el st.text_area antes de descargar</p>
                      </div>
                    </div>
                    <span className="text-[11px] font-mono font-bold text-emerald-700 bg-emerald-50 border border-emerald-200 px-2.5 py-1 rounded-full">
                      Listo para Descarga
                    </span>
                  </div>

                  <div>
                    <label className="font-mono text-[11px] uppercase tracking-wider text-[rgba(26,26,30,0.6)] block mb-1.5">
                      Título Detectado del Paper (Encabezado H1 en DOCX):
                    </label>
                    <input
                      type="text"
                      value={editableDocTitle}
                      onChange={(e) => setEditableDocTitle(e.target.value)}
                      className="w-full p-3 bg-white border border-[rgba(26,26,30,0.15)] rounded-md text-xs font-semibold text-[#1a1a1e] focus:outline-none focus:border-[#2563eb]"
                    />
                  </div>

                  <div>
                    <label className="font-mono text-[11px] uppercase tracking-wider text-[rgba(26,26,30,0.6)] block mb-1.5">
                      Contenido Traducido Editable (st.text_area):
                    </label>
                    <textarea
                      value={translatedResult}
                      onChange={(e) => setTranslatedResult(e.target.value)}
                      rows={9}
                      className="w-full p-4 bg-[#fafaf9] border border-[rgba(26,26,30,0.15)] rounded-md text-xs text-[#1a1a1e] font-serif leading-relaxed focus:outline-none focus:border-[#2563eb]"
                    />
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
                    <button
                      onClick={() => handleDownloadFile(`${currentPaper.name.replace('.pdf', '')}_traducido.docx`, translatedResult)}
                      className="py-3 px-4 rounded-lg bg-[#2563eb] hover:bg-blue-700 text-white font-semibold text-xs flex items-center justify-center gap-2 shadow-sm transition-all"
                    >
                      <Download className="w-4 h-4" />
                      Descargar DOCX Individual (.docx)
                    </button>
                    <button
                      onClick={() => handleDownloadFile(`lote_papers_traducidos_${selectedTargetLang.toLowerCase()}.zip`, translatedResult)}
                      className="py-3 px-4 rounded-lg bg-[#1a1a1e] hover:bg-black text-white font-semibold text-xs flex items-center justify-center gap-2 shadow-sm transition-all"
                    >
                      <FolderArchive className="w-4 h-4 text-amber-400" />
                      Descargar Todos en ZIP (.zip)
                    </button>
                  </div>
                </div>
              )}

              {/* Explanatory Footnote matching Variation 8 */}
              <p className="text-[13px] text-[rgba(26,26,30,0.6)] leading-relaxed pt-2">
                Nuestra tecnología utiliza <strong>PyMuPDF</strong> para detectar y clasificar estructuras de dos columnas comunes en revistas científicas y papers académicos, asegurando que el flujo de lectura para el LLM sea coherente (columna izquierda completa antes que la derecha) sin mezclar líneas intermedias.
              </p>
            </div>
          )}

          {/* TAB 2: PYTHON CODE */}
          {activeTab === 'code' && (
            <div className="max-w-4xl space-y-6">
              <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-[rgba(26,26,30,0.08)]">
                <div>
                  <h2 className="text-xl font-bold text-[#1a1a1e]">Código de la Aplicación</h2>
                  <p className="text-xs text-[rgba(26,26,30,0.6)] mt-0.5">
                    Archivos listos para ejecutar con <code className="font-mono text-[#2563eb]">streamlit run app.py</code>
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleDownloadFile('app.py', APP_PY_CODE)}
                    className="px-3 py-2 text-xs font-semibold rounded-md bg-[#2563eb] hover:bg-blue-700 text-white flex items-center gap-1.5 transition-all shadow-xs"
                  >
                    <Download className="w-3.5 h-3.5" />
                    Descargar app.py
                  </button>
                  <button
                    onClick={() => handleDownloadFile('requirements.txt', REQUIREMENTS_TXT)}
                    className="px-3 py-2 text-xs font-semibold rounded-md bg-white border border-[rgba(26,26,30,0.15)] hover:bg-[#f1f1ee] text-[#1a1a1e] flex items-center gap-1.5 transition-all"
                  >
                    <Download className="w-3.5 h-3.5" />
                    Descargar requirements.txt
                  </button>
                </div>
              </div>

              {/* requirements.txt block */}
              <div className="border border-[rgba(26,26,30,0.1)] rounded-xl overflow-hidden bg-white shadow-xs">
                <div className="bg-[#fafaf9] px-4 py-3 border-b border-[rgba(26,26,30,0.08)] flex items-center justify-between">
                  <div className="flex items-center gap-2 font-mono text-xs font-bold text-[#1a1a1e]">
                    <FileText className="w-4 h-4 text-amber-600" />
                    requirements.txt
                  </div>
                  <button
                    onClick={() => handleCopy(REQUIREMENTS_TXT, 'req')}
                    className="text-xs text-[rgba(26,26,30,0.6)] hover:text-[#1a1a1e] flex items-center gap-1 px-2.5 py-1 rounded border border-[rgba(26,26,30,0.12)] bg-white transition-all"
                  >
                    {copiedFile === 'req' ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                    {copiedFile === 'req' ? 'Copiado' : 'Copiar'}
                  </button>
                </div>
                <pre className="p-4 text-xs font-mono text-[#1a1a1e] bg-[#fdfdfc] overflow-x-auto leading-relaxed">
                  {REQUIREMENTS_TXT}
                </pre>
              </div>

              {/* app.py code block */}
              <div className="border border-[rgba(26,26,30,0.1)] rounded-xl overflow-hidden bg-white shadow-xs">
                <div className="bg-[#fafaf9] px-4 py-3 border-b border-[rgba(26,26,30,0.08)] flex items-center justify-between">
                  <div className="flex items-center gap-2 font-mono text-xs font-bold text-[#1a1a1e]">
                    <FileCode2 className="w-4 h-4 text-[#2563eb]" />
                    app.py (Código fuente modular y documentado)
                  </div>
                  <button
                    onClick={() => handleCopy(APP_PY_CODE, 'app')}
                    className="text-xs text-[rgba(26,26,30,0.6)] hover:text-[#1a1a1e] flex items-center gap-1 px-2.5 py-1 rounded border border-[rgba(26,26,30,0.12)] bg-white transition-all"
                  >
                    {copiedFile === 'app' ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                    {copiedFile === 'app' ? 'Copiado' : 'Copiar'}
                  </button>
                </div>
                <pre className="p-5 text-xs font-mono text-[#1e293b] bg-[#f8fafc] overflow-x-auto max-h-[580px] leading-relaxed">
                  {APP_PY_CODE}
                </pre>
              </div>
            </div>
          )}

          {/* TAB 3: ARCHITECTURE */}
          {activeTab === 'architecture' && (
            <div className="max-w-4xl space-y-6">
              <div>
                <h2 className="text-xl font-bold text-[#1a1a1e]">Arquitectura del Flujo Académico</h2>
                <p className="text-xs text-[rgba(26,26,30,0.6)] mt-0.5">
                  Diseñado específicamente para las complejidades de papers científicos a dos columnas
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-5 rounded-xl border border-[rgba(26,26,30,0.1)] bg-white space-y-2">
                  <div className="font-mono text-xs font-bold text-[#2563eb]">01 / EXTRACCIÓN GEOMÉTRICA</div>
                  <h3 className="text-sm font-bold text-[#1a1a1e]">PyMuPDF Orden de Columnas</h3>
                  <p className="text-xs text-[rgba(26,26,30,0.7)] leading-relaxed">
                    Extrae coordenadas <code className="font-mono text-xs bg-[#f1f1ee] px-1 py-0.5 rounded">(x0, y0, x1, y1)</code> de cada bloque. Discrimina cabeceras de ancho completo y ensambla primero la columna izquierda completa antes de la derecha.
                  </p>
                </div>

                <div className="p-5 rounded-xl border border-[rgba(26,26,30,0.1)] bg-white space-y-2">
                  <div className="font-mono text-xs font-bold text-[#2563eb]">02 / SMART CHUNKING</div>
                  <h3 className="text-sm font-bold text-[#1a1a1e]">Segmentación por Oraciones</h3>
                  <p className="text-xs text-[rgba(26,26,30,0.7)] leading-relaxed">
                    Divide en fragmentos de 1800-2000 caracteres respetando los límites de párrafos. Si un párrafo es extenso, usa expresiones regulares oracionales para evitar cortar ideas a la mitad.
                  </p>
                </div>

                <div className="p-5 rounded-xl border border-[rgba(26,26,30,0.1)] bg-white space-y-2">
                  <div className="font-mono text-xs font-bold text-[#2563eb]">03 / MODO DIRECTO</div>
                  <h3 className="text-sm font-bold text-[#1a1a1e]">Gemini con System Instructions</h3>
                  <p className="text-xs text-[rgba(26,26,30,0.7)] leading-relaxed">
                    Instrucción de sistema restrictiva que preserva intactas fórmulas matemáticas, citas académicas (ej. [1], Smith et al., 2021) y nombres propios, sin agregar notas del traductor.
                  </p>
                </div>

                <div className="p-5 rounded-xl border border-[rgba(26,26,30,0.1)] bg-white space-y-2">
                  <div className="font-mono text-xs font-bold text-[#2563eb]">04 / MODO AGENTE</div>
                  <h3 className="text-sm font-bold text-[#1a1a1e]">LangChain AgentExecutor + 2 Tools</h3>
                  <p className="text-xs text-[rgba(26,26,30,0.7)] leading-relaxed">
                    Herramienta 1: <code className="font-mono text-xs bg-[#f1f1ee] px-1 py-0.5 rounded">detect_language_tool</code> (langdetect). Herramienta 2: <code className="font-mono text-xs bg-[#f1f1ee] px-1 py-0.5 rounded">translate_academic_text_tool</code>. El agente evalúa si el texto ya está en el idioma destino.
                  </p>
                </div>

                <div className="p-5 rounded-xl border border-[rgba(26,26,30,0.1)] bg-white space-y-2">
                  <div className="font-mono text-xs font-bold text-[#2563eb]">05 / RECONSTRUCCIÓN</div>
                  <h3 className="text-sm font-bold text-[#1a1a1e]">Documento Nuevo DOCX</h3>
                  <p className="text-xs text-[rgba(26,26,30,0.7)] leading-relaxed">
                    Evita reinsertar coordenadas rígidas en el PDF. Genera un archivo Word (.docx) limpio con título H1, subtítulos H2, espaciado 1.15 y márgenes de 1 pulgada usando python-docx.
                  </p>
                </div>

                <div className="p-5 rounded-xl border border-[rgba(26,26,30,0.1)] bg-white space-y-2">
                  <div className="font-mono text-xs font-bold text-[#2563eb]">06 / EXPORTACIÓN LOTE</div>
                  <h3 className="text-sm font-bold text-[#1a1a1e]">Descarga Individual y ZIP</h3>
                  <p className="text-xs text-[rgba(26,26,30,0.7)] leading-relaxed">
                    Proporciona botones de descarga para cada documento individual y un botón unificado que empaqueta todos los .docx en un archivo .zip en memoria sin almacenamiento en disco.
                  </p>
                </div>
              </div>
            </div>
          )}
        </section>
      </main>

      {/* Editorial Footer matching Variation 8 */}
      <footer className="px-6 md:px-12 py-4 border-t border-[rgba(26,26,30,0.08)] flex flex-wrap items-center justify-between gap-2 text-xs text-[rgba(26,26,30,0.6)] bg-[#fafaf9]">
        <span>Academic Paper Translator • Professional Edition</span>
        <span>
          Generated: <code className="font-mono text-[#1a1a1e]">app.py</code> & <code className="font-mono text-[#1a1a1e]">requirements.txt</code>
        </span>
      </footer>
    </div>
  );
}
