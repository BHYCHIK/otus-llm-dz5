from langchain_huggingface import HuggingFaceEmbeddings
import torch

class Embedder:
    device: str = "cpu"
    _embedding_model_name: str = "fitlemon/bge-m3-ru-ostap"

    def __init__(self, _device="cpu", model='fitlemon/bge-m3-ru-ostap'):
        self.device = _device
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        if _device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

        self._embedding_model_name = model

        self._embedding_model = HuggingFaceEmbeddings(
            model_name=self._embedding_model_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )

    def get_embedding(self, text: str) -> list[float]:
        return self._embedding_model.embed_query(text=text)

    def get_model(self):
        return self._embedding_model