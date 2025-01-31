from typing import Dict, List, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
import json
from tqdm import tqdm
import faiss

@dataclass
class Document:
    page_content: str
    metadata: dict


class OptimizedTextSplitter:
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []

        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > self.chunk_size:
                if current_chunk:
                    chunks.append('. '.join(current_chunk) + '.')
                    overlap_size = 0
                    overlap_chunk = []
                    for s in reversed(current_chunk):
                        if overlap_size + len(s) <= self.overlap:
                            overlap_chunk.insert(0, s)
                            overlap_size += len(s)
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_length = overlap_size

            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')

        return chunks


class OptimizedVectorStore:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print("Initializing model...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents: List[Document] = []
        self.model_name = model_name
        print(f"Model initialized. Embedding dimension: {self.dimension}")

    def _process_document(self, url: str, text: str, splitter: OptimizedTextSplitter) -> List[Document]:
        chunks = splitter.split_text(text)
        return [Document(page_content=chunk, metadata={"url": url}) for chunk in chunks]

    def create_from_texts(self, data: Dict[str, str], chunk_size: int = 500, overlap: int = 100) -> None:
        print("Starting text processing...")
        splitter = OptimizedTextSplitter(chunk_size=chunk_size, overlap=overlap)

        # Process documents with progress bar
        with ThreadPoolExecutor() as executor:
            futures = []
            for url, text in tqdm(data.items(), desc="Processing documents", unit="doc"):
                futures.append(
                    executor.submit(self._process_document, url, text, splitter)
                )

            # Collect results with progress bar
            for future in tqdm(futures, desc="Collecting results", unit="doc"):
                self.documents.extend(future.result())

        print(f"Processing complete. Total documents: {len(self.documents)}")

        # Create embeddings with progress bar
        print("Creating embeddings...")
        texts = [doc.page_content for doc in self.documents]
        embeddings = []
        batch_size = 32

        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings", unit="batch"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings)

        embeddings = np.array(embeddings)
        print("Normalizing embeddings...")
        faiss.normalize_L2(embeddings)

        print("Adding to FAISS index...")
        self.index.add(embeddings.astype(np.float32))
        print(f"Index creation complete. Total vectors: {self.index.ntotal}")

    def search(self, query: str, k: int = 4) -> List[Document]:
        query_embedding = self.model.encode([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32),
            k
        )

        return [self.documents[i] for i in indices[0]]

    def save(self, directory: str) -> None:
        """Сохраняет векторное хранилище в указанную директорию"""
        print(f"Saving vector store to {directory}...")
        os.makedirs(directory, exist_ok=True)

        print("Saving FAISS index...")
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))

        print("Saving documents and configuration...")
        state = {
            'documents': self.documents,
            'model_name': self.model_name,
            'dimension': self.dimension
        }
        with open(os.path.join(directory, "state.pkl"), 'wb') as f:
            pickle.dump(state, f)
        print("Save complete!")

    @classmethod
    def load(cls, directory: str) -> 'OptimizedVectorStore':
        """Загружает векторное хранилище из указанной директории"""
        print(f"Loading vector store from {directory}...")

        print("Loading state...")
        with open(os.path.join(directory, "state.pkl"), 'rb') as f:
            state = pickle.load(f)

        print("Initializing instance...")
        instance = cls(model_name=state['model_name'])
        instance.documents = state['documents']
        instance.dimension = state['dimension']

        print("Loading FAISS index...")
        instance.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        print(f"Load complete! Total documents: {len(instance.documents)}")

        return instance


def create_optimized_vectorstore(data: Dict[str, str]) -> OptimizedVectorStore:
    print("Creating optimized vector store...")
    vectorstore = OptimizedVectorStore()
    vectorstore.create_from_texts(data)
    return vectorstore

