"""REST API for the policy assistant. Run with: uvicorn main:app --reload"""
from fastapi import FastAPI
from pydantic import BaseModel

from rag import ask

app = FastAPI(title="Kestrel Nova Ltd Policy Assistant")


class Question(BaseModel):
    question: str


class Answer(BaseModel):
    answered: bool
    answer: str
    citations: list[str]


@app.post("/ask", response_model=Answer)
def ask_endpoint(q: Question):
    return ask(q.question)
