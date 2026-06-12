import os
from dotenv import load_dotenv
from llama_index.core import query_engine
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

#Memuat file env
load_dotenv()

#insialisasi SECRET_KEY
key = os.getenv("SECRET_KEY")
model_ai = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")

#CODING 
Settings.llm = GoogleGenAI(model=model_ai, api_key=key) 

#embeding model
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

Settings.chunk_size = 128
Settings.chunk_overlap = 28

#data loader dari folder 
data = SimpleDirectoryReader("./data").load_data()

#buat index
index = VectorStoreIndex.from_documents(data)

query_engine = index.as_query_engine(similarity_top_k=2)

question = ("Berapa hari cuti tahunan yang didapat karyawan tetap?")

response = query_engine.query(question)

print(response.response)
