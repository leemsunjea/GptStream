import faiss
import numpy as np
from typing import List, Tuple
import logging

class FaissManager:
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.logger = logging.getLogger(__name__)

    def add_vectors(self, vectors: np.ndarray, ids: List[int] = None):
        """벡터를 인덱스에 추가합니다."""
        try:
            if ids is None:
                ids = np.arange(len(vectors))
            self.index.add_with_ids(vectors, np.array(ids))
            self.logger.info(f"Added {len(vectors)} vectors to FAISS index")
        except Exception as e:
            self.logger.error(f"Error adding vectors to FAISS: {str(e)}")
            raise

    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """가장 유사한 k개의 벡터를 검색합니다."""
        try:
            distances, indices = self.index.search(query_vector.reshape(1, -1), k)
            self.logger.info(f"Search completed for query vector, found {len(indices[0])} results")
            return distances[0], indices[0]
        except Exception as e:
            self.logger.error(f"Error searching in FAISS: {str(e)}")
            raise

    def save_index(self, path: str):
        """인덱스를 파일로 저장합니다."""
        try:
            faiss.write_index(self.index, path)
            self.logger.info(f"FAISS index saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving FAISS index: {str(e)}")
            raise

    def load_index(self, path: str):
        """파일에서 인덱스를 로드합니다."""
        try:
            self.index = faiss.read_index(path)
            self.logger.info(f"FAISS index loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading FAISS index: {str(e)}")
            raise 