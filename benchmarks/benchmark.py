"""
Code for reproducing llamaindex, unstructured, reducto benchmark.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import instructor
import pandas as pd
import requests
from dataset import Question, questions
from llama_parse import LlamaParse
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm.auto import tqdm
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from util import (
    SCANNED_DOCUMENT_URL,
    VectorStore,
    query_openai,
    split_pdf_into_page_chunks,
)


def create_reducto_vector_index() -> VectorStore:
    """Generate Reducto Vector Store"""
    start = time.time()

    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {os.environ['REDUCTO_API_TOKEN']}",
    }

    response = requests.post(
        "https://api.reducto.ai/chunk_url",
        headers=headers,
        params={"document_url": SCANNED_DOCUMENT_URL, "llm_table_summary": "true"},
    )

    # Download from returned download link
    download_response = requests.get(response.json())
    chunks = json.loads(download_response.text)

    print(f"Reducto took: {time.time() - start} seconds")

    return VectorStore(
        [chunk["embed"] for chunk in chunks], [chunk["raw_text"] for chunk in chunks]
    )


def create_llamaparse_vector_index() -> VectorStore:
    """Generate LlamaParse Index"""

    # Llama Parse can't handle long files, so we break them into 50 page chunks
    document_sections = split_pdf_into_page_chunks(50)

    start = time.time()
    parser = LlamaParse(
        api_key=os.environ["LLAMA_PARSE_API_KEY"], result_type="markdown"
    )

    total_docs = []
    for chunk in document_sections:
        documents = parser.load_data(chunk)
        for doc in documents:
            total_docs.append(doc.text)

    print(f"Llama parse took: {time.time() - start} seconds")

    markdown = "\n".join(total_docs)

    chunk_size = 500
    chunks = [markdown[i : i + chunk_size] for i in range(0, len(markdown), chunk_size)]

    return VectorStore(chunks)


def create_unstructured_vector_index() -> VectorStore:
    """Generate Unstructured Index"""

    # Ran into errors with documents > 50 pages on unstructured
    document_sections = split_pdf_into_page_chunks(50)

    s = UnstructuredClient(api_key_auth=os.environ["UNSTRUCTURED_API_KEY"])

    start = time.time()

    results = []  # Initialize an empty list to store results
    for fname in document_sections:
        with open(fname, "rb") as file:
            req = shared.PartitionParameters(
                files=shared.Files(
                    content=file.read(),
                    file_name=fname,
                ),
                strategy="auto",
            )
            res = s.general.partition(req)
            results.append(res.elements)

    print(f"Unstructured took {time.time() - start} seconds")

    chunk_size = 500
    full = "\n".join(["\n".join([block["text"] for block in r]) for r in results])
    chunks = [full[i : i + chunk_size] for i in range(0, len(full), chunk_size)]
    return VectorStore(chunks)


def run_rag_process(vector_store: VectorStore):
    with ThreadPoolExecutor(max_workers=20) as executor:

        def answer_question(question: Question):
            retrieved = vector_store.retrieve(
                question.query, max_context_size_chars=8000
            )
            return query_openai(
                system="You are a question answering assistant. Answer the question based on the following context:\n"
                + retrieved,
                content=question.query,
            ), retrieved

        futures = [executor.submit(answer_question, question) for question in questions]

        for _ in tqdm(
            as_completed(futures),
            total=len(questions),
        ):
            pass

        results = []
        for i, future in enumerate(futures):
            answer, retrieved = future.result()
            results.append(
                [
                    questions[i].query,
                    questions[i].answer,
                    answer,
                    retrieved,
                ]
            )
        df = pd.DataFrame(
            results,
            columns=[
                "question",
                "solution",
                "answer",
                "context",
            ],
        )

        return df


def grade_rag_results(df: pd.DataFrame):
    client = instructor.patch(OpenAI())

    class GradedQuestion(BaseModel):
        correct: bool = Field(
            ...,
            description="If the answer is correct or not. It's fine if there is extra info provided.",
        )
        explanation: str | None = Field(
            ..., description="Explanation of grading. Keep it short (max 2 sentences)."
        )

    def grade_single(question: str, answer: str, solution: str) -> GradedQuestion:
        try:
            result: GradedQuestion = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert grader, grading a response to the following question:\n{question}\n\n"
                        f"The correct answer is:\n{solution}\n\n",
                    },
                    {
                        "role": "user",
                        "content": f"Grade the following response:\n{answer}",
                    },
                ],
                response_model=GradedQuestion,
            )
        except Exception as e:
            return GradedQuestion(correct=False, explanation=f"Error grading: {e}")
        return result

    with ThreadPoolExecutor(max_workers=20) as executor:
        df_subset = df[["question", "answer", "solution"]]
        futures = [
            executor.submit(grade_single, *row)
            for row in df_subset.itertuples(index=False)
        ]

        for _ in tqdm(
            as_completed(futures),
            total=len(df),
            desc="Grading",
        ):
            pass

        grades = [future.result() for future in futures]

        df["correct"] = [g.correct for g in grades]
        df["explanation"] = [g.explanation for g in grades]

    return df


if __name__ == "__main__":
    store_fns = [
        # create_llamaparse_vector_index,
        create_reducto_vector_index,
        # create_unstructured_vector_index,
    ]

    for store_fn in store_fns:
        if os.path.exists(store_fn.__name__ + ".csv"):
            rag_df = pd.read_csv(store_fn.__name__ + ".csv")
        else:
            store = store_fn()  # swap with other parsers
            rag_df = run_rag_process(store)
            rag_df.to_csv(store_fn.__name__ + ".csv")
        grades = grade_rag_results(rag_df)
        grades.to_csv("graded" + store_fn.__name__ + ".csv")

    print(
        "Remember, GPT-4 based grading _has_ inaccuracies. Makes rue to manually review each input/output pair for grading."
    )
