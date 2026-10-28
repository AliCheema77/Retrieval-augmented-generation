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
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END, START
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
MIN_RELEVANT = 1
MAX_RETRIES = 2


class RAGState(TypedDict, total=False):
    question: str
    chat_history: list[BaseMessage]
    standalone: str
    queries: list[str]
    rankings: list[list[Document]]
    candidates: list[Document]
    docs: list[Document]
    retries: int
    answer: str


class SearchQueries(BaseModel):
    queries: list[str] = Field(description="Alternative phrasing of the question for search")


class GradeDocument(BaseModel):
    relevant: bool = Field(description="True if the document contains information useful for answering the question")


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


grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are grading whether a retrieved document is relevant to a user's question. "
         "It does not need to fully answer the question; mark it relevant if it contains "
         "information that would help answer it. Otherwise mark it not relevant."),
        ("human", "Question: {question}\n\nDocument:\n{document}"),
    ])

transform_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "A search over a document collection found nothing relevant for the question "
         "below. Rewrite it as a better search query: think about what the underlying "
         "intent is and use different, more general or more specific wording. "
         "These queries were already tried:\n{tried}\n\nReturn only the new question."),
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
            key = (doc.metadata["source"], doc.page_content)
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
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    if not vector_store.get(limit=1)["ids"]:
        vector_store.add_documents(chunks)
        print(f"Indexed {len(chunks)} chunks.")

    return vector_store


def build_graph():
    chunks = load_chunks()
    bm25_retriever = BM25Retriever.from_documents(chunks, k=6)
    vector_retriever = get_vector_store(chunks).as_retriever(search_kwargs={"k": 6})
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

    reranker = CrossEncoderReranker(
        model=HuggingFaceCrossEncoder(model_name=RERANK_MODEL),
        top_n=FINAL_K
    )

    llm = ChatOpenAI(model=CHAT_MODEL)
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    expand_chain = expand_prompt | llm.with_structured_output(SearchQueries)
    grade_chain = grade_prompt | llm.with_structured_output(GradeDocument)
    transform_chain = transform_prompt | llm | StrOutputParser()
    answer_chain = answer_prompt | llm | StrOutputParser()

    def rewrite(state: RAGState):
        standalone = rewrite_chain.invoke(
            {"question": state.get("question"), "chat_history": state.get("chat_history")}
        )
        return {"standalone": standalone}

    def expand(state: RAGState):
        standalone = state.get("standalone", state["question"])
        result = expand_chain.invoke(
            {"standalone": standalone}
        )
        return {"standalone": standalone, "queries": [standalone, *result.queries]}

    def retrieve(state: RAGState):
        return {"rankings": hybrid_retriever.batch(state["queries"])}

    def fuse(state: RAGState):
        return {"candidates": reciprocal_rank_fusion(state["rankings"])}

    def rerank(state: RAGState):
        return {"docs": reranker.compress_documents(state["candidates"], state["standalone"])}

    def grade(state: RAGState):
        grades = grade_chain.batch(
            [{"question": state["standalone"], "document": doc.page_content} for doc in state["docs"]]
        )
        relevant = [doc for doc, g in zip(state["docs"], grades) if g.relevant]
        print(f"  graded: {len(relevant)}/{len(state['docs'])} relevant")
        return {"docs": relevant}

    def transform_query(state: RAGState):
        retries = state.get("retries", 0) + 1
        new_question = transform_chain.invoke(
            {"standalone": state["standalone"], "tried": "\n".join(state["queries"])}
        )
        print(f"  retry {retries}: {new_question}")
        return {"standalone": new_question, "retries": retries}

    def generate(state: RAGState):
        answer = answer_chain.invoke(
            {
                "context": format_docs(state["docs"]),
                "chat_history": state["chat_history"],
                "question": state["question"],
            }
        )
        return {"answer": answer}

    def route_start(state: RAGState):
        return "rewrite" if state["chat_history"] else "expand"

    def route_after_grade(state: RAGState):
        if len(state["docs"]) >= MIN_RELEVANT:
            return "generate"
        if state.get("retries", 0) >= MAX_RETRIES:
            return "generate"
        return "transform_query"

    graph = StateGraph(RAGState)

    graph.add_node("rewrite", rewrite)
    graph.add_node("expand", expand)
    graph.add_node("retrieve", retrieve)
    graph.add_node("fuse", fuse)
    graph.add_node("rerank", rerank)
    graph.add_node("grade", grade)
    graph.add_node("transform_query", transform_query)
    graph.add_node("generate", generate)

    graph.add_conditional_edges(START, route_start, ["rewrite", "expand"])
    graph.add_edge("rewrite", "expand")
    graph.add_edge("expand", "retrieve")
    graph.add_edge("retrieve", "fuse")
    graph.add_edge("fuse", "rerank")
    graph.add_edge("rerank", "grade")
    graph.add_conditional_edges("grade", route_after_grade, ["generate", "transform_query"])
    graph.add_edge("transform_query", "expand")
    graph.add_edge("generate", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    app.get_graph().print_ascii()

    chat_history = []
    while True:
        query = input("Ask your questions here (or type 'exit' to quit)\n")
        if query.lower() == "exit":
            break
        result = app.invoke({"question": query, "chat_history": chat_history})

        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=result["answer"]))

        sources = sorted({Path(doc.metadata["source"]).name for doc in result["docs"]})
        print("\n(searched for:)")
        for q in result["queries"]:
            print(f"  - {q}")
        print(f"\nAnswer:\n{result['answer']}\n\nSources: {', '.join(sources) or 'none'}\n")


