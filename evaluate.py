# SSL bypass — debe ir antes de cualquier otro import
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

"""
evaluate.py — Script de evaluación de métricas del RAGSystem.

Métricas calculadas:
  - Recall@5:    ¿Aparece el documento correcto entre los 5 recuperados? (0 o 1 por pregunta)
  - Precision@5: ¿Qué fracción de los 5 recuperados proviene del documento correcto?

Lógica de evaluación:
  Para cada pregunta en el golden_set.json:
    1. Se recuperan los top-5 fragmentos con el RAGSystem (híbrido BM25 + Pinecone).
    2. Se verifica cuántos provienen del archivo fuente esperado.
    3. Recall@5  = 1 si al menos 1 coincide, 0 si ninguno coincide.
    4. Precision@5 = (fragmentos del archivo esperado) / 5

  Se promedian ambas métricas sobre las 5 preguntas del golden set.

Uso:
    python evaluate.py
"""

import json
from pathlib import Path
from rag_system import RAGSystem


def cargar_golden_set(ruta: str = "golden_set.json") -> list[dict]:
    """Carga el conjunto de preguntas y respuestas esperadas."""
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def calcular_metricas(
    recuperados: list,
    documento_esperado: str,
    k: int = 5,
) -> tuple[int, float]:
    """
    Calcula Recall@k y Precision@k para una sola pregunta.

    Args:
        recuperados:        Lista de Document recuperados por el sistema.
        documento_esperado: Nombre del archivo fuente correcto (ej. "tortas_y_pasteles.md").
        k:                  Número de documentos a considerar.

    Returns:
        (recall, precision): Recall es 0 o 1; Precision es 0.0 a 1.0.
    """
    top_k = recuperados[:k]

    # Contar cuántos fragmentos provienen del documento esperado
    hits = sum(
        1
        for doc in top_k
        if documento_esperado in doc.metadata.get("source", "")
    )

    recall = 1 if hits > 0 else 0
    precision = hits / k if k > 0 else 0.0

    return recall, precision


def evaluar(golden_set: list[dict], sistema: RAGSystem, k: int = 5) -> dict:
    """
    Ejecuta la evaluación completa sobre el golden set.

    Args:
        golden_set: Lista de dicts con 'pregunta' y 'documento_id_esperado'.
        sistema:    Instancia de RAGSystem ya inicializada.
        k:          Top-k a evaluar.

    Returns:
        Dict con resultados detallados y métricas promedio.
    """
    resultados = []

    print(f"\n{'='*60}")
    print(f"  EVALUACIÓN — Recall@{k} y Precision@{k}")
    print(f"{'='*60}")

    for i, item in enumerate(golden_set, 1):
        pregunta = item["pregunta"]
        esperado = item["documento_id_esperado"]

        print(f"\n[{i}/{len(golden_set)}] {pregunta}")
        print(f"  → Esperado: {esperado}")

        # Recuperar documentos
        recuperados = sistema.retrieve(pregunta)

        # Mostrar qué se recuperó
        fuentes_recuperadas = [doc.metadata.get("source", "?") for doc in recuperados]
        print(f"  → Recuperados: {fuentes_recuperadas}")

        # Calcular métricas
        recall, precision = calcular_metricas(recuperados, esperado, k)

        estado_recall = "✅" if recall == 1 else "❌"
        print(f"  → Recall@{k}: {recall} {estado_recall}  |  Precision@{k}: {precision:.2f}")

        resultados.append({
            "pregunta": pregunta,
            "documento_esperado": esperado,
            "fuentes_recuperadas": fuentes_recuperadas,
            "recall": recall,
            "precision": precision,
        })

    # Promedios
    recall_promedio = sum(r["recall"] for r in resultados) / len(resultados)
    precision_promedio = sum(r["precision"] for r in resultados) / len(resultados)

    return {
        "detalle": resultados,
        "recall_promedio": recall_promedio,
        "precision_promedio": precision_promedio,
        "k": k,
        "n_preguntas": len(resultados),
    }


def imprimir_reporte(metricas: dict) -> None:
    """Imprime el resumen final de la evaluación."""
    k = metricas["k"]
    n = metricas["n_preguntas"]
    recall = metricas["recall_promedio"]
    precision = metricas["precision_promedio"]

    print(f"\n{'='*60}")
    print(f"  RESUMEN DE EVALUACIÓN")
    print(f"{'='*60}")
    print(f"  Preguntas evaluadas : {n}")
    print(f"  Top-k               : {k}")
    print(f"")
    print(f"  Recall@{k}  promedio : {recall:.2%}")
    print(f"  Precision@{k} promedio: {precision:.2%}")
    print(f"{'='*60}")
    print()
    print("  ¿Qué significan estos números?")
    print()
    print(f"  Recall@{k} = {recall:.2%}")
    print(f"    En el {recall:.0%} de las preguntas, el documento correcto")
    print(f"    apareció entre los {k} fragmentos recuperados.")
    print()
    print(f"  Precision@{k} = {precision:.2%}")
    print(f"    En promedio, el {precision:.0%} de los {k} fragmentos recuperados")
    print(f"    provenía del archivo fuente correcto.")
    print()

    # Interpretación cualitativa
    if recall >= 0.8:
        print("  📈 Recall alto: el sistema encuentra el documento correcto con frecuencia.")
    elif recall >= 0.6:
        print("  📊 Recall moderado: el sistema falla en algunos casos. Revisá el chunking.")
    else:
        print("  📉 Recall bajo: el sistema pierde información relevante. Revisá embeddings.")

    if precision >= 0.4:
        print("  🎯 Precision aceptable: los resultados son mayormente relevantes.")
    else:
        print("  ⚠️  Precision baja: el sistema recupera mucho ruido junto con lo relevante.")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("  PASTELERÍA RAG — Script de Evaluación")
    print("=" * 60)

    # Cargar golden set
    golden_set = cargar_golden_set("golden_set.json")
    print(f"\n📋 Golden set cargado: {len(golden_set)} preguntas.")

    # Inicializar el sistema (carga BM25 + conecta Pinecone)
    sistema = RAGSystem(top_k=5)

    # Evaluar
    metricas = evaluar(golden_set, sistema, k=5)

    # Imprimir reporte
    imprimir_reporte(metricas)
