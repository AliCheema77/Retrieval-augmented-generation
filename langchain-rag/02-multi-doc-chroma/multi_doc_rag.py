from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

SCRIPT_DIR = Path(__file__).parent
DOCS_DIR = SCRIPT_DIR / "documents"
CHROMA_DIR = SCRIPT_DIR / "chroma_store"
CHAT_MODEL = "gpt-6-luna"
EMBEDDING_MODEL = "text-embedding-3-small"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a helpful assistant. Answer the user's question using ONLY the "
         "context below. Each chunk starts with its source file in brackets. "
         "If the answer isn't in the context, say you don't know.\n\n"
         "Context:\n{context}"),
        ("human", "{question}"),
    ]
)


def format_docs(docs):
    return "\n\n----\n\n".join(
        f"[{Path(doc.metadata['source']).name}]\n{doc.page_content}" for doc in docs
    )


def get_vector_store():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name="documents",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    # Only embed on the first run. After that the vectors are already on disk.
    # If you change the documents, delete chroma_store/ to rebuild.
    if not vector_store.get(limit=1)["ids"]:
        loader = DirectoryLoader(
            str(DOCS_DIR),
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        vector_store.add_documents(chunks)
        print(f"Indexed {len(chunks)} chunks form {len(documents)} files.")

    return vector_store


def build_chain():
    retriever = get_vector_store().as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model=CHAT_MODEL)

    answer_chain = (
            RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
            | prompt
            | llm
            | StrOutputParser()
    )

    # Keep the raw docs in the output so we can show sources next to the answer.
    return (
            {"docs": retriever, "question": RunnablePassthrough()}
            | RunnablePassthrough.assign(answer=answer_chain)
    )


if __name__ == "__main__":
    chain = build_chain()
    print(chain.get_graph().print_ascii() )
    while True:
        query = input("Ask your questions here (or type 'exit')\n")
        if query.lower() == "exit":
            break
        result = chain.invoke(query)
        sources = sorted({Path(doc.metadata['source']).stem for doc in result["docs"]})
        print(f"\nAnswer:\n{result['answer']}\n\nSources: {', '.join(sources)}\n")

