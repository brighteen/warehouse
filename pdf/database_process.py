from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings

load_dotenv()

def create_vector_store(collection_name, db_path, passage_embeddings=UpstageEmbeddings(model="solar-embedding-1-large-passage")):
    """
    백터 스토어를 생성합니다.
    """
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=passage_embeddings,
        persist_directory=db_path,
    )
    return vector_store

def add_documents(vector_store, documents):
    all_data = vector_store.get()
    start = len(all_data["ids"]) + 1
    end = len(documents) + len(all_data["ids"]) + 1
    uuids = [str(i) for i in range(start, end)]
    vector_store.add_documents(documents=documents, ids=uuids)
    return vector_store

def select_docs(db, query):
    print(f"[디버그] db : {db}")
    retriever = db.as_retriever()
    selected_docs = retriever.invoke(query)
    return selected_docs
