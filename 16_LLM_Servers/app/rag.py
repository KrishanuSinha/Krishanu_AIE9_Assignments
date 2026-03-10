"""Provider-aware RAG helpers for the Assignment 16 cat-health application."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Annotated, TypedDict

import langsmith as ls
import tiktoken
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore

from app.models import Provider, get_chat_model, get_embedding_model


def _tiktoken_len(text: str) -> int:
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(text)
    return len(tokens)


class RAGRunResult(TypedDict):
    provider: str
    question: str
    answer: str
    retrieved_contexts: list[str]
    source_documents: list[dict[str, str | int | None]]
    usage_metadata: dict


@lru_cache(maxsize=1)
def _get_source_documents() -> tuple[Document, ...]:
    data_dir = os.environ.get("RAG_DATA_DIR", "data")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"RAG data directory not found: {data_dir}")

    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
    )
    documents = loader.load()
    if not documents:
        raise RuntimeError(f"No PDF documents found under: {data_dir}")
    return tuple(documents)


@lru_cache(maxsize=1)
def _get_chunks() -> tuple[Document, ...]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=100,
        length_function=_tiktoken_len,
    )
    return tuple(splitter.split_documents(list(_get_source_documents())))


@lru_cache(maxsize=2)
def _get_vectorstore(provider: Provider) -> QdrantVectorStore:
    return QdrantVectorStore.from_documents(
        documents=list(_get_chunks()),
        embedding=get_embedding_model(provider=provider),
        location=":memory:",
        collection_name=f"rag_collection_{provider}",
    )


def warm_rag_indexes() -> None:
    """Build both vector indexes outside tracing so setup cost is not mixed with query cost."""
    _get_vectorstore("fireworks")
    if os.environ.get("OPENAI_API_KEY"):
        _get_vectorstore("openai")


def _get_usage_metadata(message: AIMessage) -> dict:
    if getattr(message, "usage_metadata", None):
        return dict(message.usage_metadata)

    response_metadata = getattr(message, "response_metadata", {}) or {}
    token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
    return dict(token_usage or {})


def _serialize_sources(docs: list[Document]) -> list[dict[str, str | int | None]]:
    payload: list[dict[str, str | int | None]] = []
    for doc in docs:
        payload.append(
            {
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "chunk_preview": doc.page_content[:180],
            }
        )
    return payload


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a veterinary RAG assistant. Answer only from the supplied context. "
            'If the context does not contain the answer, say "I don\'t know based on the document."',
        ),
        (
            "human",
            "QUESTION:\n{question}\n\nCONTEXT:\n{context}",
        ),
    ]
)


@ls.traceable(run_type="chain", name="rag_app_query")
def answer_question(question: str, provider: Provider = "fireworks") -> RAGRunResult:
    retriever = _get_vectorstore(provider).as_retriever(
        search_kwargs={"k": int(os.environ.get("RAG_TOP_K", "4"))}
    )
    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    llm = get_chat_model(provider=provider, temperature=0)
    ai_msg = (PROMPT | llm).invoke(
        {
            "question": question,
            "context": context_text,
        }
    )

    return {
        "provider": provider,
        "question": question,
        "answer": ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content),
        "retrieved_contexts": [doc.page_content for doc in retrieved_docs],
        "source_documents": _serialize_sources(retrieved_docs),
        "usage_metadata": _get_usage_metadata(ai_msg),
    }


@tool
def retrieve_information(
    query: Annotated[str, "query to ask the retrieve information tool"],
) -> str:
    """Use RAG over the feline care guideline PDF."""
    return answer_question(query, provider="fireworks")["answer"]
