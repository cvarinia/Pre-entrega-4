# 🥐 Pastelería RAG — Sistema escalable con Pinecone

Sistema RAG con recuperación híbrida para el asistente de pastelería. Extiende la Pre-entrega 3 reemplazando ChromaDB local por **Pinecone Serverless** en la nube y agregando un **Recuperador Híbrido** que combina búsqueda semántica con búsqueda léxica (BM25).

---

## Arquitectura

```
Documentos .md
      ↓
  [RecursiveCharacterTextSplitter]  chunk_size=1000 chars
      ↓
  [Google embedding-001] → vectores de 768 dimensiones
      ↓
  [Pinecone Serverless] ← namespace="menu-pasteleria"
                            metadata: text, source, categoria, chunk_id

               ┌─────────────┐     ┌──────────────┐
Consulta →    │ BM25Retriever│  +  │PineconeRetriever│
               │  (léxico)   │     │  (semántico)  │
               └──────┬──────┘     └──────┬────────┘
                      └────────┬──────────┘
                     [EnsembleRetriever]
                     (Reciprocal Rank Fusion)
                      peso BM25=0.4, Pinecone=0.6
                               ↓
                          Top-5 resultados
```

---

## Estructura del proyecto

```
pasteleria-pinecone/
├── data/
│   ├── tortas_y_pasteles.md      # Ingredientes y alérgenos de tortas
│   ├── facturas_y_masas.md       # Facturas, medialunas, scones
│   ├── alergenos_y_dietas.md     # Tabla de alérgenos y guía del personal
│   └── bebidas_y_extras.md       # Bebidas y adicionales
├── ingest.py                     # Pipeline de ingesta a Pinecone
├── rag_system.py                 # Clase RAGSystem con EnsembleRetriever
├── evaluate.py                   # Métricas Recall@5 y Precision@5
├── golden_set.json               # 5 preguntas con respuestas conocidas
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Requisitos previos

- Python 3.10 o superior
- **Google AI Studio API Key** (gratuita): https://aistudio.google.com/app/apikey
- **Pinecone API Key** (plan Starter gratuito): https://app.pinecone.io/

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/pasteleria-pinecone.git
cd pasteleria-pinecone

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus claves reales
```

---

## Cómo replicar el índice de Pinecone

### Paso 1 — Crear la cuenta en Pinecone

1. Registrarse en https://app.pinecone.io/ (plan Starter gratuito)
2. Ir a **API Keys** y copiar la clave
3. Pegarla en el `.env` como `PINECONE_API_KEY`

### Paso 2 — Correr la ingesta

```bash
python ingest.py
```

El script hace automáticamente:
- Verifica si el índice `pasteleria-rag` existe (lo crea si no)
- Configura dimensión=768 (Google embedding-001), metric=cosine, AWS us-east-1
- Verifica si el namespace ya tiene vectores (evita duplicados)
- Carga y fragmenta los 4 archivos .md
- Sube los vectores con metadatos: `text`, `source`, `categoria`, `chunk_id`

Salida esperada:
```
⚙️  Creando índice 'pasteleria-rag'...
📂 Cargando documentos desde './data'...
   → 4 documentos cargados.
   → 22 fragmentos creados
⬆️  Subiendo a Pinecone (namespace='menu-pasteleria')...
✅ Ingesta completada. 22 vectores en namespace 'menu-pasteleria'.
```

### Paso 3 — Probar el sistema

```bash
python rag_system.py
```

Corre una consulta de demo y muestra qué retriever encontró cada resultado (BM25, Pinecone, o ambos).

### Paso 4 — Evaluar métricas

```bash
python evaluate.py
```

---

## Resultados de evaluación

Métricas obtenidas con el golden set de 5 preguntas sobre alérgenos y dietas:

| Métrica | Valor |
|---|---|
| **Recall@5** | Ver consola al correr `evaluate.py` |
| **Precision@5** | Ver consola al correr `evaluate.py` |

### ¿Qué mide cada métrica?

**Recall@5**: De todas las preguntas evaluadas, ¿en qué porcentaje apareció el documento correcto entre los 5 recuperados?
- 100% = el sistema siempre encuentra la fuente correcta
- 0% = el sistema no encontró ninguna fuente correcta

**Precision@5**: Del total de 5 fragmentos recuperados por pregunta, ¿qué fracción provenía del documento correcto?
- 1.0 = todos los 5 fragmentos vienen de la fuente correcta
- 0.2 = solo 1 de 5 fragmentos es relevante

---

## Decisiones de diseño

| Decisión | Valor | Justificación |
|---|---|---|
| Vector DB | Pinecone Serverless | Escalable, sin gestión de infraestructura |
| Embeddings | Google embedding-001 | Gratuito, 768 dims, mismo modelo en ingesta y consulta |
| Dimensión | 768 | Requerida por Google embedding-001 |
| top_k | 5 | Evita "Lost in the Middle" sin perder cobertura |
| BM25 weight | 0.4 | Léxico útil para nombres técnicos exactos |
| Semantic weight | 0.6 | Semántico domina para preguntas en lenguaje natural |
| Namespace | "menu-pasteleria" | Permite aislar datos si se agrega más contenido |
| text en metadata | Sí | Evita consultas adicionales a una base relacional |

---

## Variables de entorno requeridas

| Variable | Descripción |
|---|---|
| `GOOGLE_API_KEY` | API Key de Google AI Studio |
| `PINECONE_API_KEY` | API Key de Pinecone |
| `PINECONE_INDEX_NAME` | Nombre del índice (default: `pasteleria-rag`) |
