from lime.lime_text import LimeTextExplainer
import numpy as np
from sentence_transformers import SentenceTransformer

class BERTSimilarityExplainer:
    def __init__(self, base_description):
        self.base_description = base_description

        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.base_embedding = self.model.encode([base_description], convert_to_numpy=True)[0]

        def similarity_func(texts):
            emb = self.model.encode(texts, convert_to_numpy=True)
           
            sim = np.dot(emb, self.base_embedding) / (np.linalg.norm(emb, axis=1) * np.linalg.norm(self.base_embedding) + 1e-10)
            # LIME用に2次元配列に変換([Not Similar, Similar]の２次元)

            return np.vstack([1 - sim, sim]).T

        self.similarity_score = similarity_func

        self.explainer = LimeTextExplainer(
            class_names=["Not Similar", "Similar"],
            bow=True,
            verbose=False,
            random_state=42,
            feature_selection='none',
            kernel_width=25,
        )

    def explain(self, target_description, num_features=10):
        explanation = self.explainer.explain_instance(
            target_description,
            self.similarity_score,
            labels=[1],
            num_features=num_features,
            num_samples=100,  # デフォルトは5000　かなり減らして高速化している
        )
        return explanation
