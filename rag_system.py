"""
rag_system.py — Sistema RAG con Recuperador Híbrido (BM25 + Pinecone).

La clase RAGSystem combina dos tipos de búsqueda:
  - Búsqueda semántica (vectorial): Pinecone con embeddings de Google.
  - Búsqueda léxica (BM25): ranking por frecuencia de términos, ideal para
    palabras técnicas exactas como "torta de zanahoria" o "sin TACC".

Los resultados de ambas se fusionan con Reciprocal Rank Fusion (RRF),
dando pesos configurables a cada retriever.

Uso desde código:
    from rag_system import RAGSystem
    sistema = RAGSystem()
    documentos = sistema.retrieve("¿La torta de zanahoria tiene gluten?")
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from pinecone import Pinecone
from langchain_community.retrievers import BM25Retriever
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

DATA_DIR = "./data"
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pasteleria-rag")
INDEX_HOST = os.getenv("PINECONE_INDEX_HOST", "pasteleria-rag-2ug33wx.svc.aped-4627-b74a.pinecone.io")
NAMESPACE = "menu-pasteleria"
TOP_K = 5


def get_embeddings() -> HuggingFaceEmbeddings:
    """Mismo modelo que en ingest.py — crítico para que las distancias sean coherentes."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )


def cargar_documentos_locales() -> list[Document]:
    """
    Carga los documentos .md de /data para el BM25Retriever.

    BM25 es un retriever local que trabaja en memoria, por eso necesita
    los documentos cargados. Los mismos que se subieron a Pinecone.
    """
    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    return loader.load()


class RAGSystem:
    """
    Sistema de recuperación híbrida que combina búsqueda vectorial y léxica.

    Atributos:
        top_k (int): Número de documentos a recuperar (por defecto 5).
        bm25_weight (float): Peso del retriever BM25 en el ensemble (0.0 a 1.0).
        semantic_weight (float): Peso del retriever de Pinecone (0.0 a 1.0).

    Nota: bm25_weight + semantic_weight deben sumar 1.0.
    """

    def __init__(
        self,
        top_k: int = TOP_K,
        bm25_weight: float = 0.4,
        semantic_weight: float = 0.6,
    ):
        self.top_k = top_k
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self._setup()

    def _setup(self):
        """Inicializa los dos retrievers."""

        # --- Retriever 1: BM25 (léxico / keyword matching) ---
        print("🔤 Inicializando BM25Retriever (búsqueda léxica)...")
        documentos = cargar_documentos_locales()
        self.bm25_retriever = BM25Retriever.from_documents(documentos)
        self.bm25_retriever.k = self.top_k

        # --- Retriever 2: Pinecone (semántico / vectorial) ---
        # Con fallback a BM25 si hay error SSL de red
        self.pinecone_retriever = None
        try:
            print("🧠 Inicializando PineconeVectorStore (búsqueda semántica)...")
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index(host=INDEX_HOST)
            vectorstore = PineconeVectorStore(
                index=index,
                embedding=get_embeddings(),
                namespace=NAMESPACE,
            )
            self.pinecone_retriever = vectorstore.as_retriever(
                search_kwargs={"k": self.top_k}
            )
            print("✅ Pinecone conectado correctamente.")
        except Exception as e:
            print(f"⚠️  Pinecone no disponible por error de red/SSL: {type(e).__name__}")
            print("   → El sistema continúa usando solo BM25 (léxico).")

        print(
            f"✅ RAGSystem listo "
            f"(BM25={self.bm25_weight}, Semántico={self.semantic_weight}, top_k={self.top_k})"
        )

    def retrieve(self, query: str) -> list[Document]:
        """
        Recibe una consulta y devuelve los top-k documentos más relevantes.
        Fusión manual con Reciprocal Rank Fusion (RRF), equivalente al EnsembleRetriever.

        RRF: cada doc recibe score = weight / (60 + rank). Los docs que aparecen
        bien en ambas listas acumulan más puntaje y suben al tope.
        """
        bm25_docs = self.bm25_retriever.invoke(query)

        # Si Pinecone no está disponible, retornar solo BM25
        if self.pinecone_retriever is None:
            return bm25_docs[: self.top_k]

        pinecone_docs = self.pinecone_retriever.invoke(query)

        doc_scores: dict[str, float] = {}
        doc_objects: dict[str, Document] = {}

        for rank, doc in enumerate(pinecone_docs):
            key = doc.page_content[:120]
            doc_scores[key] = doc_scores.get(key, 0.0) + self.semantic_weight / (60 + rank + 1)
            doc_objects[key] = doc

        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content[:120]
            doc_scores[key] = doc_scores.get(key, 0.0) + self.bm25_weight / (60 + rank + 1)
            if key not in doc_objects:
                doc_objects[key] = doc

        sorted_keys = sorted(doc_scores, key=lambda k: doc_scores[k], reverse=True)
        return [doc_objects[k] for k in sorted_keys[: self.top_k]]

    def retrieve_con_scores(self, query: str) -> list[tuple[Document, str]]:
        """
        Versión extendida que muestra el origen de cada resultado (BM25, Pinecone o ambos).
        """
        bm25_docs = self.bm25_retriever.invoke(query)
        pinecone_docs = self.pinecone_retriever.invoke(query)
        ensemble_docs = self.retrieve(query)

        bm25_keys = {d.page_content[:120] for d in bm25_docs}
        pinecone_keys = {d.page_content[:120] for d in pinecone_docs}

        resultados = []
        for doc in ensemble_docs:
            key = doc.page_content[:120]
            en_bm25 = key in bm25_keys
            en_pinecone = key in pinecone_keys
            if en_bm25 and en_pinecone:
                origen = "BM25 + Pinecone ⭐"
            elif en_bm25:
                origen = "BM25"
            else:
                origen = "Pinecone"
            resultados.append((doc, origen))

        return resultados


# ---------------------------------------------------------------------------
# Demo rápido
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sistema = RAGSystem()

    query = "¿Tienen opciones para personas con alergia al gluten?"
    print(f"\n🔍 Consulta: {query}\n")

    resultados = sistema.retrieve_con_scores(query)
    for i, (doc, origen) in enumerate(resultados, 1):
        print(f"  [{i}] {origen}")
        print(f"      Fuente: {doc.metadata.get('source', '?')}")
        print(f"      Texto:  {doc.page_content[:120].strip()}...")
        print()
