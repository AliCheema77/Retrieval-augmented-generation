from os import path
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai.types import vector_store

# load_dotenv() searches upward from this file's folder, so it finds the
# .env at the repo root and puts OPENAI_API_KEY into the environment,
# where the langchain_openai classes pick it up automatically.
load_dotenv()


SCRIPT_DIR = Path(__file__).parent
CHAT_MODEL = "gpt-6-luna"
EMBEDDING_MODEL = "text-embedding-3-small"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a helpful assistant. Answer the user's question using ONLY the "
         "context below. If the answer isn't in the context, say you don't know.\n\n"
         "Context:\n{context}"),
        ("human", "{question}"),
    ])


def format_docs(docs):
    return "\n\n----\n\n".join(doc.page_content for doc in docs)


def build_chain():
    path = SCRIPT_DIR / "resume.txt"
    documents = [Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": str(path)})]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    vector_store = InMemoryVectorStore.from_documents(chunks, OpenAIEmbeddings(model=EMBEDDING_MODEL))
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model=CHAT_MODEL)

    # LCEL: `|` pipes each step's output into the next. The dict runs both
    # branches on the same input question: one retrieves and formats context,
    # the other passes the question through unchanged. Together they fill
    # the {context} and {question} slots in the prompt.

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

if __name__ == "__main__":
      chain = build_chain()
      print(chain.get_graph().print_ascii() )
      while True:
          query = input("Ask your questions here (or type 'exit' to quit)\n")
          if query.lower() == "exit":
              break
          print(f"\nAnswer:\n{chain.invoke(query)}\n")