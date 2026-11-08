from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

SCRIPT_DIR = Path(__file__).parent
DOCS_DIR = SCRIPT_DIR / "documents"
CHROMA_DIR = SCRIPT_DIR / "chroma_store"
CHAT_MODEL = "gpt-6-luna"
EMBEDDING_MODEL = "text-embedding-3-small"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FINAL_K = 4

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about a collection of "
    "documents (such as a resume and a company handbook). Use the search_documents "
    "tool to look things up; never answer questions about the documents from memory. "
    "If the first results aren't enough, search again with different wording, or "
    "make separate searches for separate parts of the question. If the information "
    "isn't in the documents after searching, say you don't know. For greetings and "
    "small talk, reply directly without searching. Mention the source file names "
    "your answer is based on."
)


def format_docs(docs):
    return "\n\n----\n\n".join(
        f"[{Path(doc.metadata['source']).name}]\n{doc.page_content}" for doc in docs
    )


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
        weights=[0.5, 0.5],
    )
    reranker = CrossEncoderReranker(
        model=HuggingFaceCrossEncoder(model_name=RERANK_MODEL),
        top_n=FINAL_K,
    )

    # The docstring is what the LLM reads to decide when to call this tool.
    # content_and_artifact: the string goes to the LLM, the Document list is
    # kept on the ToolMessage (.artifact) so our code can show sources.
    @tool(response_format="content_and_artifact")
    def search_documents(query: str):
        """Search the document collection and return the most relevant passages,
        each labelled with its source file. Use a short, specific search query."""
        candidates = hybrid_retriever.invoke(query)
        docs = reranker.compress_documents(candidates, query)
        return format_docs(docs), docs

    tools = [search_documents]
    llm = ChatOpenAI(model=CHAT_MODEL).bind_tools(tools)

    def agent(state: MessagesState):
        response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    app = build_graph()
    app.get_graph().print_ascii()

    config = {"configurable": {"thread_id": "chat-1"}, "recursion_limit": 15}

    while True:
        query = input("Ask your questions here (or type 'exit' to quit)\n")
        if query.lower() == "exit":
            break

        before = len(app.get_state(config).values.get("messages", []))
        result = app.invoke({"messages": [HumanMessage(content=query)]}, config)
        new_messages = result["messages"][before:]

        sources = set()
        for message in new_messages:
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    print(f"  searched: {call['args']['query']}")
            if isinstance(message, ToolMessage):
                sources.update(Path(doc.metadata["source"]).name for doc in message.artifact)

        print(f"\nAnswer:\n{result['messages'][-1].text}\n\nSources: {', '.join(sorted(sources)) or 'none'}\n")