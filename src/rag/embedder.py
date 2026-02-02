from langchain_huggingface import HuggingFaceEmbeddings
import torch

class Embedder:
    device: str = "cpu"
    embedding_model_name: str = "fitlemon/bge-m3-ru-ostap"

    def __init__(self, device="cpu", model='fitlemon/bge-m3-ru-ostap'):
        self.device = device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

        self.embedding_model_name = model

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )

    def get_embedding(self, text: str) -> list[float]:
        return self.embedding_model.embed_query(text=text)