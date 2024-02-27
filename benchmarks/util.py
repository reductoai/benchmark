from typing import List, Callable
import os
import pickle
import hashlib
import numpy as np
from openai import OpenAI, RateLimitError
import pandas as pd
from tqdm.auto import tqdm
import concurrent.futures
from PyPDF2 import PdfReader, PdfWriter
import requests
import backoff

client = OpenAI()


@backoff.on_exception(backoff.expo, RateLimitError)
def query_openai(system: str, content: str) -> str:
    result = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        temperature=0.1,
        seed=1,
    )
    return result.choices[0].message.content


SCANNED_DOCUMENT_URL = (
    "https://utfs.io/f/c2d22275-9c66-4a0b-bbc5-dad333043d76-c2t3e8.pdf"
)


def generate_embeddings(text: str) -> np.ndarray:
    """Generate embeddings with OpenAI"""
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return np.array(response.data[0].embedding)


def split_pdf_into_page_chunks(chunk_size: int) -> List[str]:
    if not os.path.exists(".cache/uber-scan.pdf"):
        os.makedirs(".cache", exist_ok=True)
        with requests.get(SCANNED_DOCUMENT_URL, stream=True) as r:
            r.raise_for_status()
            with open(".cache/uber-scan.pdf", "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    reader = PdfReader(".cache/uber-scan.pdf")
    writer = PdfWriter()

    total_pages = len(reader.pages)
    output_files = []

    for i in range(0, total_pages, chunk_size):
        writer = PdfWriter()  # Create a new PdfWriter instance for each chunk
        for page in range(i, min(i + chunk_size, total_pages)):
            writer.add_page(reader.pages[page])

        output_filename = f".cache/scanned_uber_chunk_{i//chunk_size + 1}.pdf"
        with open(output_filename, "wb") as output_file:
            writer.write(output_file)
        output_files.append(output_filename)

    return output_files


class VectorStore:
    """Maintains a local vector store for a set of strings"""

    def __init__(
        self,
        chunks: List[str],
        chunk_values: List[str] | None = None,
        embed_fn: Callable[[str], np.ndarray] = generate_embeddings,
    ) -> None:
        self.chunks = chunks
        self.chunk_values = chunk_values
        self.embed_fn = embed_fn
        self.cache_dir = ".cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        self.chunks_hash = hashlib.sha256("".join(chunks).encode()).hexdigest()

        self.cache_path = os.path.join(
            self.cache_dir, f"vector_store_{self.chunks_hash}.pkl"
        )

        if self.chunk_values is not None:
            self.cache_path = self.cache_path.replace(
                "vector_store", "vector_store_values_new"
            )

        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as cache_file:
                self.store = pickle.load(cache_file)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                futures = [
                    executor.submit(self.embed_fn, chunk) for chunk in self.chunks
                ]
                for _ in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Embedding chunks",
                ):
                    pass
                embeddings = [future.result() for future in futures]
            if chunk_values is not None:
                assert len(chunk_values) == len(chunks)
                self.store = pd.DataFrame(
                    {"chunk": chunks, "value": chunk_values, "embed": embeddings}
                )
            else:
                self.store = pd.DataFrame({"chunk": self.chunks, "embed": embeddings})
            with open(self.cache_path, "wb") as cache_file:
                pickle.dump(self.store, cache_file)

    def retrieve(
        self, query: str, max_context_size_chars: int, separator=f"\n{'=' * 16}\n"
    ) -> str:
        query_embedding = self.embed_fn(query)
        self.store["similarity"] = self.store["embed"].apply(
            lambda x: np.dot(x, query_embedding)
            / (np.linalg.norm(x) * np.linalg.norm(query_embedding))
        )
        sorted = self.store.sort_values("similarity", ascending=False)[
            "chunk" if self.chunk_values is None else "value"
        ].tolist()

        ret = ""
        for i, s in enumerate(sorted):
            if len(s) + len(ret) > max_context_size_chars:
                break
            if i != 0:
                ret += separator
            ret += s
        return ret
