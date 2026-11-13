"""RAG pipeline: load policies, chunk by section, embed, retrieve, answer with citations."""
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pydantic import BaseModel, Field

load_dotenv()

# Folder next to this file, so it works whichever folder you start the server from
POLICY_DIR = Path(__file__).parent / "policies"


# ---------- 1. Load and chunk ----------
def load_chunks(folder=POLICY_DIR):
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "policy"), ("##", "section")]
    )
    chunks = []
    for f in sorted(folder.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        text = re.sub(r"^>.*FICTIONAL.*\n", "", text)  # drop disclaimer
        meta = re.search(r"Policy (\S+) · Version (\S+)", text)
        for chunk in splitter.split_text(text):
            if "section" not in chunk.metadata:  # skip title/metadata-only chunk
                continue
            chunk.metadata.update(
                source=f.name,
                policy_id=meta.group(1) if meta else None,
                version=meta.group(2) if meta else None,
                chunk_id=f"{f.name}#{chunk.metadata['section']}",
            )
            chunk.page_content = (
                f"{chunk.metadata['policy']} > {chunk.metadata['section']}\n\n{chunk.page_content}"
            )
            chunks.append(chunk)
    return chunks


# ---------- 2. Embed and store (runs once, when this file is imported) ----------
chunks = load_chunks()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)


# ---------- 3. Answer format, labels, prompt ----------
class PolicyAnswer(BaseModel):
    answered: bool = Field(description="True only if the sources contain the answer")
    answer: str = Field(description="The answer, naming the policy it comes from. If not answered, say the policies do not cover this.")
    citations: list[str] = Field(description="Required when answered is true: the source labels in [brackets] that you used, copied exactly. Empty if not answered.")


def label(d):
    return f"{d.metadata['policy_id']} | {d.metadata['section']}"


def format_context(docs):
    return "\n\n".join(f"[{label(d)}]\n{d.page_content}" for d in docs)


SYSTEM_PROMPT = """You are an HR assistant for Kestrel Nova Ltd that answers employee questions about company policies.

Rules:
- Answer using ONLY the policy context provided. Do not use outside knowledge.
- If the sources do not contain the answer, set answered to false and say the policies do not cover it. Do not guess.
- If the sources only partly answer the question, answer the part they cover and say what is not covered.
- Whenever answered is true, you must list in the citations field every source label (shown in [brackets]) you used, copied exactly.

Style:
- Write in plain, conversational English, as if speaking to a colleague. Address the employee as "you".
- Explain the answer in your own words; do not copy sentences from the policy.
- Include any conditions, deadlines or exceptions that affect the answer.
- Use plain text only: no Markdown, no asterisks, no bullet points.
- Do not put source labels in the answer text; list them only in the citations field.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Question: {question}\n\nSources:\n{context}"),
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = prompt | llm.with_structured_output(PolicyAnswer)


# ---------- 4. Ask ----------
FALLBACK = "The policy documents provided do not contain an answer to this question."


def ask(question, k=4, min_score=0.4):
    results = vectorstore.similarity_search_with_score(question, k=k)

    # Guard 1: nothing relevant retrieved, so don't call the LLM at all
    if not results or results[0][1] < min_score:
        return {"answered": False, "answer": FALLBACK, "citations": []}

    docs = [doc for doc, _ in results]
    result = chain.invoke({"context": format_context(docs), "question": question})

    # The LLM itself says it's not covered: use its wording
    if not result.answered:
        return {"answered": False, "answer": result.answer, "citations": []}

    # Guard 2: it claims an answer but cites nothing real, so treat the answer as unsafe
    retrieved_labels = {label(d) for d in docs}
    valid = [c for c in result.citations if c in retrieved_labels]
    if not valid:
        return {"answered": False, "answer": FALLBACK, "citations": []}

    return {"answered": True, "answer": result.answer, "citations": valid}
