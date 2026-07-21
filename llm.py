from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate 
import os


def get_pdf():
    loader = PyPDFLoader("운수.pdf")
    pages = loader.load_and_split()
    return pages


def pdf_split():
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200
    )
    
    pages = get_pdf()
    texts = text_splitter.split_documents(pages)
    return texts


def get_embedding():
    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
    return embeddings


def get_vectorDB():
    texts = pdf_split()
    embeddings = get_embedding()

    if os.path.exists("chroma_db"):
        db = Chroma(persist_directory="chroma_db", embedding_function=embeddings, collection_name='chroma-db')
    else:
        db = Chroma.from_documents(documents=texts, embedding=embeddings, collection_name='chroma-db',persist_directory="chroma_db")
    return db


def get_ai_message(user_question):
    llm = ChatUpstage(model="solar-pro")
    db = get_vectorDB()
    retriever = db.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages([
    ("system", """
    - 다음 문맥(context)을 참고하여 질문에 답변하세요.
    - 문맥에 직접적으로 명시되지 않았더라도, 문맥에 있는 정보로부터 합리적으로 추론할 수 있다면 추론해서 답변하세요.
    - 예를 들어 인물이 특정 대상을 간절히 원하거나 반복적으로 언급한다면, 이는 그 인물이 그것을 좋아하거나 원한다는 근거로 볼 수 있습니다.
    - 문맥과 전혀 관련이 없거나 추론조차 불가능한 질문일 때만 "모른다"고 답변하세요.
    {context}"""),
    ("human", "{input}")
])

    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    ai_message = retrieval_chain.invoke({"input": user_question})
    return ai_message["answer"]
