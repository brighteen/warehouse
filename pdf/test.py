from langchain_upstage import UpstageEmbeddings
from database_process import create_vector_store

# UpstageEmbeddings 객체를 한 번만 생성합니다.
upstage = UpstageEmbeddings(model="solar-embedding-1-large-passage")

# 임베딩 함수를 래핑하여 callable하게 만듭니다.
embedding_function = lambda texts: upstage.embed_texts(texts)

# 벡터 스토어를 생성할 때, passage_embeddings 인자로 embedding_function을 전달합니다.
vector_store = create_vector_store(
    collection_name="example_collection",
    db_path="pdf/chroma_langchain_db",
    passage_embeddings=embedding_function
)

print(vector_store)