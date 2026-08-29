import logging
from collections.abc import Callable
from time import perf_counter
from typing import TypedDict

from app.config import Settings, get_settings
from chains.answer_chain import generate_answer
from kg.neo4j_store import KnowledgeGraphStore
from langgraph.graph import END, START, StateGraph
from rag.retriever import build_vector_hits, format_vector_hits
from rag.vector_store import get_embeddings, get_vector_store


logger = logging.getLogger(__name__)


class RagState(TypedDict, total=False):
    question: str
    query_embedding: list[float]
    embedding_dimensions: int
    vector_context: list[str]
    vector_hits: list[dict[str, object]]
    graph_context: list[str]
    answer: str


ProgressCallback = Callable[[str, str, dict[str, object]], None]


def build_rag_graph(
    settings: Settings | None = None,
    progress: ProgressCallback | None = None,
):
    settings = settings or get_settings()
    embeddings = get_embeddings(
        settings.embedding_model,
        settings.embedding_api_key,
        settings.embedding_base_url,
    )
    vector_store = get_vector_store(
        settings.chroma_persist_dir,
        settings.embedding_model,
        settings.embedding_api_key,
        settings.embedding_base_url,
    )
    kg_store = None

    def report(stage: str, status: str, **detail: object) -> None:
        if progress is not None:
            progress(stage, status, detail)

    def encode_question_node(state: RagState) -> dict[str, object]:
        report("encode", "running", model=settings.embedding_model)
        started = perf_counter()
        embedding = embeddings.embed_query(state["question"])
        duration_ms = round((perf_counter() - started) * 1000)
        report(
            "encode",
            "completed",
            model=settings.embedding_model,
            dimensions=len(embedding),
            duration_ms=duration_ms,
            vector_preview=[round(value, 6) for value in embedding[:8]],
        )
        return {
            "query_embedding": embedding,
            "embedding_dimensions": len(embedding),
        }

    def retrieve_vector_node(state: RagState) -> dict[str, object]:
        report("vector_search", "running", top_k=settings.retrieval_top_k)
        started = perf_counter()
        results = vector_store.similarity_search_by_vector_with_relevance_scores(
            state["query_embedding"],
            k=settings.retrieval_top_k,
        )
        hits = build_vector_hits(results)
        duration_ms = round((perf_counter() - started) * 1000)
        report(
            "vector_search",
            "completed",
            top_k=settings.retrieval_top_k,
            matches=len(hits),
            duration_ms=duration_ms,
            vector_hits=hits,
        )
        return {
            "vector_context": format_vector_hits(hits),
            "vector_hits": hits,
        }

    def retrieve_graph_node(state: RagState) -> dict[str, list[str]]:
        nonlocal kg_store
        report("graph_search", "running", limit=settings.graph_result_limit)
        started = perf_counter()
        try:
            kg_store = kg_store or KnowledgeGraphStore(settings)
            context = kg_store.search_context(
                state["question"],
                limit=settings.graph_result_limit,
            )
            report(
                "graph_search",
                "completed",
                matches=len(context),
                duration_ms=round((perf_counter() - started) * 1000),
            )
            return {
                "graph_context": context
            }
        except Exception as exc:
            # ponytail: Neo4j enriches answers but is not required; keep vector QA usable.
            logger.warning("Knowledge graph retrieval unavailable: %s", exc)
            report(
                "graph_search",
                "degraded",
                matches=0,
                duration_ms=round((perf_counter() - started) * 1000),
                message="Neo4j 不可用，已降级为纯向量检索",
            )
            return {"graph_context": []}

    def generate_answer_node(state: RagState) -> dict[str, str]:
        report("answer", "running", model=settings.llm_model)
        started = perf_counter()
        answer = generate_answer(
            question=state["question"],
            vector_context=state["vector_context"],
            graph_context=state["graph_context"],
            model_name=settings.llm_model,
            openai_api_key=settings.openai_api_key,
            openai_base_url=settings.openai_base_url,
        )
        report(
            "answer",
            "completed",
            model=settings.llm_model,
            duration_ms=round((perf_counter() - started) * 1000),
        )
        return {"answer": answer}

    workflow = StateGraph(RagState)
    workflow.add_node("encode_question", encode_question_node)
    workflow.add_node("retrieve_vector", retrieve_vector_node)
    workflow.add_node("retrieve_graph", retrieve_graph_node)
    workflow.add_node("generate_answer", generate_answer_node)

    workflow.add_edge(START, "encode_question")
    workflow.add_edge(START, "retrieve_graph")
    workflow.add_edge("encode_question", "retrieve_vector")
    workflow.add_edge(["retrieve_vector", "retrieve_graph"], "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()
