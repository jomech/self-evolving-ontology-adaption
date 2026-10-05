import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


# ontology_agent.py is inside EvoOntology/scripts.
# parents[1] is the EvoOntology project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(
    dotenv_path=ENV_PATH,
    override=True,
)

ONTOLOGY_PATH = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else str(PROJECT_ROOT / "outputs" / "ontology_v0.json")
)

QUESTION = (
    sys.argv[2]
    if len(sys.argv) > 2
    else "What is the main method and how is it evaluated?"
)

MODEL = os.getenv("GROQ_MODEL")
API_KEY = os.getenv("GROQ_API_KEY")

def tokenize(text):
    return set(
        re.findall(
            r"[a-zA-Z0-9_]+",
            text.lower(),
        )
    )


def search_ontology(ontology, query, limit=12):
    query_tokens = tokenize(query)
    results = []

    for group in [
        "terms",
        "evidence",
        "mappings",
        "constraints",
        "relations",
    ]:
        for record in ontology.get(group, []):
            text = json.dumps(
                record,
                ensure_ascii=False,
            ).lower()

            record_tokens = tokenize(text)
            overlap = query_tokens.intersection(
                record_tokens
            )

            if overlap:
                results.append(
                    {
                        "kind": group,
                        "score": len(overlap),
                        "record": record,
                    }
                )

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:limit]


def extract_text(response):
    choices = getattr(response, "choices", None)

    if not choices:
        raise RuntimeError(
            "Groq returned no choices."
        )

    choice = choices[0]
    message = getattr(choice, "message", None)

    if message is None:
        raise RuntimeError(
            "Groq response has no message."
        )

    content = getattr(message, "content", None)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)

            elif isinstance(block, dict):
                value = (
                    block.get("text")
                    or block.get("content")
                )

                if isinstance(value, str):
                    parts.append(value)

        return "\n".join(parts).strip()

    raise RuntimeError(
        f"Unsupported Groq content type: "
        f"{type(content).__name__}"
    )


def main():
    if not API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing from .env"
        )

    if not MODEL:
        raise RuntimeError(
            "GROQ_MODEL is missing from .env"
        )

    if not ONTOLOGY_PATH.exists():
        raise FileNotFoundError(
            f"Ontology not found: {ONTOLOGY_PATH}"
        )

    ontology = json.loads(
        ONTOLOGY_PATH.read_text(
            encoding="utf-8"
        )
    )

    matches = search_ontology(
        ontology=ontology,
        query=QUESTION,
        limit=12,
    )

    semantic_context = json.dumps(
        matches,
        indent=2,
        ensure_ascii=False,
    )

    system_prompt = """You are an ontology-assisted PDF research agent.

Use the supplied ontology records as your semantic index.
Answer only from the evidence and records provided.
Do not invent facts.
Mention page numbers when available.
If the ontology is insufficient, say exactly what is missing.
"""

    user_prompt = f"""Question:
{QUESTION}

Relevant ontology records:
{semantic_context}

Return:
1. Direct answer
2. Supporting ontology records
3. Evidence pages
4. Missing information or uncertainty
"""

    client = Groq(api_key=API_KEY)

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=1000,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    answer = extract_text(response)

    trajectory = {
        "question": QUESTION,
        "ontology_version": ontology.get(
            "metadata",
            {},
        ).get(
            "ontology_version",
            "v0",
        ),
        "retrieved_records": matches,
        "answer": answer,
    }

    Path("outputs").mkdir(exist_ok=True)

    Path(
        "outputs/trajectory_001.json"
    ).write_text(
        json.dumps(
            trajectory,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + answer)
    print(
        "\nSaved: outputs/trajectory_001.json"
    )


if __name__ == "__main__":
    main()