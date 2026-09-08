"""
ingest.py — Pipeline de ingesta de documentos hacia Pinecone Serverless.

Qué hace:
1. Verifica si el índice existe en Pinecone y lo crea si no.
2. Verifica si el namespace ya tiene datos (evita re-indexar y gastar cuota).
3. Carga archivos .md de /data y los fragmenta con RecursiveCharacterTextSplitter.
4. Genera embeddings con Google (768 dims) y los sube a Pinecone.
5. Incluye metadatos avanzados: texto completo, fuente, categoría y chunk_id.

Uso:
    python ingest.py
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

# --- Configuración ---
DATA_DIR = "./data"
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pasteleria-rag")
NAMESPACE = "menu-pasteleria"
DIMENSION = 768          # Google models/embedding-001 produce vectores de 768 dims
METRIC = "cosine"

# Chunking: ~250 chars ≈ 60 tokens (adecuado para fichas de producto cortas)
# Para documentos técnicos más largos, usar 2000-3000 chars ≈ 500-800 tokens
CHUNK_SIZE = 1000        # caracteres
CHUNK_OVERLAP = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )


def inferir_categoria(filename: str) -> str:
    """Asigna una categoría basada en el nombre del archivo fuente."""
    nombre = Path(filename).stem.lower()
    if "torta" in nombre or "pastel" in nombre:
        return "tortas-pasteles"
    if "factura" in nombre or "masa" in nombre:
        return "facturas-masas"
    if "alergen" in nombre or "dieta" in nombre:
        return "alergenos-dietas"
    if "bebida" in nombre or "extra" in nombre:
        return "bebidas-extras"
    return "general"


# ---------------------------------------------------------------------------
# Setup de Pinecone
# ---------------------------------------------------------------------------

def setup_pinecone_index(pc: Pinecone) -> None:
    """
    Verifica si el índice existe. Si no, lo crea como Serverless en AWS us-east-1.
    Espera a que esté listo antes de continuar.
    """
    existing = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME in existing:
        print(f"✅ Índice '{INDEX_NAME}' ya existe. Saltando creación.")
        return

    print(f"⚙️  Creando índice '{INDEX_NAME}' (dim={DIMENSION}, metric={METRIC})...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    # Esperar a que el índice esté listo
    print("   Esperando que el índice esté disponible", end="", flush=True)
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        print(".", end="", flush=True)
        time.sleep(2)
    print(" ✓")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def ingest_documents() -> None:
    """
    Carga, fragmenta y sube documentos a Pinecone.
    Detecta si el namespace ya tiene datos para evitar duplicados.
    """
    # 1. Inicializar Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    setup_pinecone_index(pc)

    # 2. Verificar si el namespace ya tiene vectores indexados
    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    namespace_count = stats.namespaces.get(NAMESPACE, {}).get("vector_count", 0)

    if namespace_count > 0:
        print(f"✅ Namespace '{NAMESPACE}' ya tiene {namespace_count} vectores. Saltando ingesta.")
        print("   Para re-indexar: eliminá todos los vectores del namespace en el panel de Pinecone.")
        return

    # 3. Cargar documentos de /data
    print(f"\n📂 Cargando documentos desde '{DATA_DIR}'...")
    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load()

    if not documents:
        raise ValueError(f"No se encontraron archivos .md en '{DATA_DIR}'.")

    print(f"   → {len(documents)} documentos cargados.")

    # 4. Fragmentar (chunking)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"   → {len(chunks)} fragmentos creados (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    # 5. Enriquecer metadatos
    # Estrategia: guardar el texto en metadata para no necesitar base relacional adicional.
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "desconocido")
        chunk.metadata.update({
            "text": chunk.page_content,          # texto completo en metadata
            "source": Path(source).name,          # solo el nombre del archivo
            "categoria": inferir_categoria(source),
            "chunk_id": f"chunk-{i:04d}",
        })

    # 6. Generar embeddings y subir a Pinecone
    print(f"\n⬆️  Subiendo a Pinecone (namespace='{NAMESPACE}')...")
    print("   Generando embeddings con Google embedding-001 (768 dims)...")

    embeddings = get_embeddings()
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace=NAMESPACE,
    )

    # 7. Verificar resultado
    stats_final = index.describe_index_stats()
    total = stats_final.namespaces.get(NAMESPACE, {}).get("vector_count", 0)
    print(f"\n✅ Ingesta completada. {total} vectores en namespace '{NAMESPACE}'.")


if __name__ == "__main__":
    print("=" * 55)
    print("  PASTELERÍA RAG — Pipeline de Ingesta a Pinecone")
    print("=" * 55)
    ingest_documents()
