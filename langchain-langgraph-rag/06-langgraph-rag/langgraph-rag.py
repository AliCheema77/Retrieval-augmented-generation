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


