# <코드 4> document_preprocessor.py
import openai

def embed_document(text, api_key):
    openai.api_key = api_key
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    vector = response["data"][0]["embedding"]
    return vector

def process_and_add_document(rag_system, doc_id, text, api_key):
    # 예: 전처리(필요하다면 토큰화, 청소 등) 후 임베딩
    embedding = embed_document(text, api_key)
    rag_system.add_document(doc_id, text, embedding)
