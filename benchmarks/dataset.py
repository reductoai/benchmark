import requests
from dataclasses import dataclass


@dataclass
class Question:
    query: str
    answer: str
    context: str


response = requests.get(
    "https://github.com/run-llama/llama-datasets/raw/8fcdc64c19d9d61981c92e7d66982bc57bdc42b8/llama_datasets/10k/uber_2021/rag_dataset.json"
)
dataset = response.json()

questions = [
    Question(
        query=q["query"],
        context="\n".join(q["reference_contexts"]),
        answer=q["reference_answer"],
    )
    for q in dataset["examples"]
]
