import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL, FAISS_INDEX_PATH
from .db_handler import db_handler
import uuid

class SemanticSearchManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SemanticSearchManager, cls).__new__(cls)
            cls._instance.model = SentenceTransformer(EMBEDDING_MODEL)
            cls._instance.dimension = cls._instance.model.get_sentence_embedding_dimension()
            cls._instance.user_indexes = {} # In-memory cache for user indexes
            if not os.path.exists(FAISS_INDEX_PATH):
                os.makedirs(FAISS_INDEX_PATH)
            print("Semantic Search Manager initialized.")
        return cls._instance

    def _get_user_index(self, user_id: str):
        if user_id in self.user_indexes:
            return self.user_indexes[user_id]

        index_file = os.path.join(FAISS_INDEX_PATH, f"{user_id}.index")
        if os.path.exists(index_file):
            index = faiss.read_index(index_file)
            # Load corresponding memory_ids
            meta_file = os.path.join(FAISS_INDEX_PATH, f"{user_id}.meta")
            with open(meta_file, 'r') as f:
                memory_ids = [line.strip() for line in f.readlines()]
        else:
            index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
            memory_ids = []
        
        self.user_indexes[user_id] = (index, memory_ids)
        return index, memory_ids

    def _save_user_index(self, user_id: str):
        if user_id not in self.user_indexes:
            return
        
        index, memory_ids = self.user_indexes[user_id]
        index_file = os.path.join(FAISS_INDEX_PATH, f"{user_id}.index")
        faiss.write_index(index, index_file)

        meta_file = os.path.join(FAISS_INDEX_PATH, f"{user_id}.meta")
        with open(meta_file, 'w') as f:
            f.write('\n'.join(memory_ids))

    def add_memory_embedding(self, user_id: str, memory_id: str, text: str):
        index, memory_ids = self._get_user_index(user_id)
        embedding = self.model.encode([text], convert_to_tensor=False)
        
        # Use a hash of memory_id as the Faiss ID to ensure uniqueness and reproducibility
        # Faiss requires integer IDs.
        faiss_id = np.array([abs(hash(memory_id)) % (2**63 - 1)], dtype=np.int64)

        index.add_with_ids(embedding, faiss_id)
        memory_ids.append(memory_id) # This is a simple mapping, for more complex scenarios, a better mapping is needed.
        # This implementation simply appends, a robust version would ensure no duplicates
        # and handle removals correctly. Let's assume for now IDs are unique.
        
        self._save_user_index(user_id)
        print(f"Added embedding for memory {memory_id} for user {user_id}")

    def search_relevant_memories(self, user_id: str, query_text: str, k: int = 5) -> list[str]:
        index, memory_ids = self._get_user_index(user_id)
        if index.ntotal == 0:
            return []
            
        query_embedding = self.model.encode([query_text], convert_to_tensor=False)
        distances, faiss_ids = index.search(query_embedding, k)
        
        # Now we need to map faiss_ids back to our memory_ids
        # This is a simplification. A production system would use a more robust mapping.
        # Let's rebuild a lookup for this search.
        id_lookup = {abs(hash(mem_id)) % (2**63 - 1): mem_id for mem_id in memory_ids}
        
        retrieved_mem_ids = []
        for i in faiss_ids[0]:
            if i in id_lookup:
                retrieved_mem_ids.append(id_lookup[i])
        
        return retrieved_mem_ids

# Singleton instance
semantic_search_manager = SemanticSearchManager()
