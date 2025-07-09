from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document

def build_vectorstore(file_path):
    # 读取知识库文本
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    # 切分为小段
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]
    # 构建向量库
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def search_knowledge(query, vectorstore, top_k=2):
    docs = vectorstore.similarity_search(query, k=top_k)
    return "\n".join([doc.page_content for doc in docs]) 