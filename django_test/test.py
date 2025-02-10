from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings
from dotenv import load_dotenv

from database_process import create_vector_store, add_documents
import processed_documents 

load_dotenv()

collection_name = "example_collection"
db_path = "./chroma_langchain_db" 
passage_embeddings = UpstageEmbeddings(model="solar-embedding-1-large-passage")

vector_store = create_vector_store(collection_name, db_path, passage_embeddings)

file_path = "data\Don't Do RAG.pdf"
texts = processed_documents.main(file_path=file_path)

vector_store = add_documents(vector_store, texts)

all_data = vector_store.get()

from llm_process import generate_response
response = generate_response(vector_store, "Explain the Rag system")

print(response)