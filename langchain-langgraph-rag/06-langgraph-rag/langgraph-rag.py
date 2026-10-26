from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from ollama import embeddings
from pydantic import BaseModel, Field

load_dotenv()

SCRIPT_DIR = Path(__file__).parent
DOCS_DIR = SCRIPT_DIR / "documents"
CHROMA_DIR = SCRIPT_DIR / "chroma_store"
CHAT_MODEL = "gpt-6-luna"
EMBEDDING_MODEL = "text-embedding-3-small"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NUM_VARIANTS = 3
CANDIDATES_K = 10
FINAL_K = 4


# total=False: the caller only passes question + chat_history; nodes fill in the rest.
class RAGState(TypedDict, total=False):
    question: str
    chat_history: list[BaseMessage]
    standalone: str
    queries: list[str]
    rankings: list[list[Document]]
    candidates: list[Document]
    docs: list[Document]
    answer: str


class SearchQueries(BaseModel):
    queries: list[str] = Field(description="Alternative phrasings of the question for search")


rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Given the chat history and the user's latest question, rewrite the question "
         "so it can be understood without the chat history. Resolve pronouns like "
         "'he', 'it', 'that' to what they refer to. Do NOT answer it. If it is "
         "already standalone, return it unchanged. Return only the question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])

expand_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         f"Write {NUM_VARIANTS} different search queries that would help find "
         "information to answer the user's question in a document collection. "
         "Use different wording and synonyms, and approach it from different angles."),
        ("human", "{standalone}"),
    ])

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a helpful assistant. Answer the user's question using ONLY the "
         "context below. Each chunk starts with its source file in brackets. "
         "If the answer isn't in the context, say you don't know.\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])


def format_docs(docs):
    return "\n\n----\n\n".join(
        f"[{Path(doc.metadata['source']).name}]\n{doc.page_content}" for doc in docs
    )


def reciprocal_rank_fusion(rankings, k=60, top_n=CANDIDATES_K):
    scores = {}
    docs_by_key = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            key = (doc.metadata['source'], doc.page_content)
            docs_by_key[key] = doc
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)

        best = sorted(scores, key=scores.get, reverse=True)[:top_n]
        return [docs_by_key[key] for key in best]


def load_chunks():
    loader = DirectoryLoader(
        str(DOCS_DIR),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(loader.load())


def get_vector_store(chunks):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name="documents",
        embedding_func=embeddings,
        persist_directory=str(CHROMA_DIR)
    )


