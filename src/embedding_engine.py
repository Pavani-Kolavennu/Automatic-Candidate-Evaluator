from sentence_transformers import SentenceTransformer


class EmbeddingEngine:

    _model = None

    def __init__(self):
        if EmbeddingEngine._model is None:
            print("Loading embedding model...")
            EmbeddingEngine._model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            print("Embedding model loaded!")

        self.model = EmbeddingEngine._model

    def encode_text(self, text):
        return self.model.encode(text, normalize_embeddings=True)

    def encode_texts(self, texts, batch_size=2048):
        return self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )

    def similarity_from_embeddings(self, emb1, emb2):
        return float(emb1 @ emb2)

    def similarities_to_embedding(self, texts, other_embedding, batch_size=2048):
        embeddings = self.encode_texts(texts, batch_size=batch_size)

        return embeddings @ other_embedding

    def similarity(self, text1, text2):
        emb1 = self.model.encode(text1, normalize_embeddings=True)
        emb2 = self.model.encode(text2, normalize_embeddings=True)

        return float(emb1 @ emb2)