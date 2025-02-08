# <코드 3> rag_pipeline.py
import chromadb
# from chromadb.config import Settings
from chromadb import PersistentClient

import os

class RAGPipeline:
    def __init__(self, db_path, openai_api_key):
        self.db_path = db_path
        self.openai_api_key = openai_api_key
        self._init_db()
        self._init_llm()

    # def _init_db(self):
    #     # Chromadb 설정
    #     self.chroma_client = chromadb.Client(Settings(
    #         chroma_db_impl="duckdb+parquet",
    #         persist_directory=self.db_path
    #     ))
    #     # 없으면 만들고 있으면 불러옴
    #     self.collection = self.chroma_client.get_or_create_collection("my_documents")

    def _init_db(self):
        # 기존처럼 persist_directory를 쓸 수 없고,
        # 0.6.x 버전에서는 PersistentClient(path=...)로 선언해야 함.
        self.chroma_client = PersistentClient(path=self.db_path)
        self.collection = self.chroma_client.get_or_create_collection("my_documents")

    def _init_llm(self):
        # 여기서는 OpenAI API 키만 등록해두고,
        # 실제 콜할 때 openai 라이브러리로 호출할 수 있음
        # (중복 import를 피하려고 여기에선 import하지 않아도 됨)
        self.model_api_key = self.openai_api_key

    def generate_answer(self, query):
        # 1) 벡터 DB에서 관련 문서 검색
        results = self.collection.query(query_texts=[query], n_results=3)
        context_docs = results["documents"][0] if results["documents"] else []

        # 2) 문서 내용을 하나로 합치기
        combined_context = "\n".join(context_docs)

        # 3) LLM에게 보낼 프롬프트 구성
        prompt = f"질의: {query}\n\n참고 문서:\n{combined_context}\n\n답변:"
        
        # 4) OpenAI API 콜 (가상의 예시)
        # 실제로는 openai 라이브러리 사용
        # ------------------------------------------------------------
        # import openai
        # openai.api_key = self.model_api_key
        # response = openai.Completion.create(
        #     model="text-davinci-003",
        #     prompt=prompt,
        #     max_tokens=512
        # )
        # answer = response["choices"][0]["text"]
        # ------------------------------------------------------------
        
        # 여기서는 그냥 가상의 응답 리턴
        answer = f"[가상 답변] 문서 기반 답변: {combined_context[:50]}..."
        return answer

    def add_document(self, doc_id, text, embedding):
        # 문서를 컬렉션에 추가
        self.collection.add(documents=[text], ids=[doc_id], embeddings=[embedding])

    def delete_document(self, doc_id):
        # 문서를 컬렉션에서 삭제
        self.collection.delete(ids=[doc_id])
