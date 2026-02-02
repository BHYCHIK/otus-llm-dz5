from langchain_huggingface import HuggingFaceEmbeddings
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

#EMBEDDING_MODEL = "BAAI/bge-m3" #квантифицированная
EMBEDDING_MODEL = "fitlemon/bge-m3-ru-ostap" #зафайнтюненная для русского языка BAAI/bge-m3

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True} # Важно для косинусного расстояния!
)

def get_embedding(text: str) -> list[float]:
    return embeddings.embed_query(text=text)