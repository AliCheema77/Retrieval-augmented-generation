from lib2to3.fixes.fix_input import context
from operator import itemgetter
from pathlib import Path
from unittest import result

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

SCRIPT_DIR = Path(__file__).parent
DOCS_DIR = SCRIPT_DIR / "documents"
CHROMA_DIR = SCRIPT_DIR / "chroma_store"
CHAT_MODEL = "gpt-6-luna"
EMBEDDING_MODEL = "text-embedding-3-small"


rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
           "Given the chat history and the user's latest question, rewrite the question "                                                                                                                                    
           "so it can be understood without the chat history. Resolve pronouns like "                                                                                                                                        
           "'he', 'it', 'that' to what they refer to. Do NOT answer it. If it is "                                                                                                                                           
           "already standalone, return it unchanged. Return only the question."),
          MessagesPlaceholder("chat_history"),
          ("human", "{question}"),
    ]
)

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
           "You are a helpful assistant. Answer the user's question using ONLY the "                                                                                                                                         
           "context below. Each chunk starts with its source file in brackets. "                                                                                                                                             
           "If the answer isn't in the context, say you don't know.\n\n"                                                                                                                                                     
           "Context:\n{context}"),
          MessagesPlaceholder("chat_history"),
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
        print(f"Indexed {len(chunks)} chunks from {len(documents)} files.")

    return vector_store


def build_chain():
    retriever = get_vector_store().as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model=CHAT_MODEL)

    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    # A function used as a step may return a runnable; LangChain then invokes
    # that runnable with the same input. This lets us skip the rewrite LLM call
    # on the first question, when there's no history to resolve.
    def standalone_question(x):
        if not x["chat_history"]:
            return x["question"]
        return rewrite_chain

    answer_chain = (
            RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
            | answer_prompt
            | llm
            | StrOutputParser()
    )

    return (
            RunnablePassthrough.assign(standalone=standalone_question)
            | RunnablePassthrough.assign(docs=itemgetter("standalone") | retriever)
            | RunnablePassthrough.assign(answer=answer_chain)
    )


if __name__ == "__main__":
    chain = build_chain()
    chat_history = []
    while True:
        query = input("Ask your questions here (or type 'exit' to exit): \n")
        if query == "exit":
            break
        result = chain.invoke({"question": query, "chat_history": chat_history})
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=result["answer"]))

        sources = sorted({Path(doc.metadata["source"]).name for doc in result["docs"]})
        print(f"\n(searched for: {result['standalone']})")
        print(f"\nAnswer:\n{result['answer']}\n\nSources: {', '.join(sources)}\n")