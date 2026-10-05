import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader


# ============================================================
# Configuration
# ============================================================

load_dotenv()

PDF_PATH = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "data/company_report.pdf"
)

OUTPUT_DIR = Path("outputs")
ONTOLOGY_PATH = OUTPUT_DIR / "ontology_v0.json"
RUN_PATH = OUTPUT_DIR / "ontology_run.json"

MODEL = os.getenv("GROQ_MODEL")
API_KEY = os.getenv("GROQ_API_KEY")

# Conservative values for a free Groq organization.
# Reduce further if your account reports a smaller TPM limit.
MAX_CHUNK_CHARS = 7_000
MAX_OUTPUT_TOKENS = 700

MAX_RETRIES = 3
RETRY_SECONDS = 10
BETWEEN_CHUNKS_SECONDS = 3


# ============================================================
# Startup validation
# ============================================================

if not API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing. Add it to the .env file."
    )

if not MODEL:
    raise RuntimeError(
        "GROQ_MODEL is missing. Add an available model ID to .env."
    )

if not PDF_PATH.exists():
    raise FileNotFoundError(
        f"PDF not found: {PDF_PATH}"
    )


# ============================================================
# JSON and response utilities
# ============================================================

def clean_json_text(text):
    """
    Clean model output before json.loads().
    """

    if not isinstance(text, str):
        raise TypeError(
            f"Expected model text as str, got {type(text).__name__}"
        )

    text = text.strip()

    # Remove <think>...</think> blocks if present.
    if "<think>" in text and "</think>" in text:
        text = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.DOTALL,
        ).strip()

    # Remove Markdown code fences.
    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # Keep only the outermost JSON object if extra text exists.
    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        text = text[start:end + 1]

    return text.strip()


def extract_response_text(response):
    """
    Extract generated text from Groq Chat Completions responses.

    Normal response:
        response.choices.message.content[0]

    This also supports:
    - content represented as a list of blocks
    - dictionary-style compatible responses
    - fallback text fields
    """

    choices = getattr(response, "choices", None)

    if choices is None and isinstance(response, dict):
        choices = response.get("choices")

    if not choices:
        raise RuntimeError(
            f"Groq returned no choices. Raw response: {response!r}"
        )

    # choices is a list.
    choice = choices[0]

    # Object-style SDK response.
    message = getattr(choice, "message", None)

    # Dictionary-style response.
    if message is None and isinstance(choice, dict):
        message = choice.get("message")

    if message is not None:
        content = getattr(message, "content", None)

        if content is None and isinstance(message, dict):
            content = message.get("content")

        # Standard string response.
        if isinstance(content, str) and content.strip():
            return content.strip()

        # List-style content response.
        if isinstance(content, list):
            parts = []

            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                    continue

                if isinstance(block, dict):
                    value = (
                        block.get("text")
                        or block.get("content")
                    )

                    if isinstance(value, str):
                        parts.append(value)

                    continue

                value = getattr(block, "text", None)

                if isinstance(value, str):
                    parts.append(value)

            if parts:
                return "\n".join(parts).strip()

    # Fallback fields on the choice.
    fallback_fields = (
        "text",
        "content",
        "output_text",
        "reasoning_content",
    )

    for field in fallback_fields:
        value = getattr(choice, field, None)

        if value is None and isinstance(choice, dict):
            value = choice.get(field)

        if isinstance(value, str) and value.strip():
            return value.strip()

    # Fallback fields on the response object.
    for field in fallback_fields:
        value = getattr(response, field, None)

        if isinstance(value, str) and value.strip():
            return value.strip()

    raise RuntimeError(
        "Could not extract generated text from Groq response. "
        f"Raw response: {response!r}"
    )


# ============================================================
# Groq request
# ============================================================

def call_groq(client, system_prompt, user_prompt):
    """
    Send one JSON-mode request to Groq and return a Python dictionary.
    """

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                max_tokens=MAX_OUTPUT_TOKENS,
                response_format={
                    "type": "json_object"
                },
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

            raw_text = extract_response_text(response)
            cleaned_text = clean_json_text(raw_text)

            try:
                parsed = json.loads(cleaned_text)
            except json.JSONDecodeError as json_error:
                raise RuntimeError(
                    "Groq returned non-JSON output.\n"
                    f"JSON error: {json_error}\n"
                    f"Returned text:\n{raw_text[:3000]}"
                ) from json_error

            if not isinstance(parsed, dict):
                raise RuntimeError(
                    "Groq returned valid JSON, but it was not a JSON object."
                )

            return parsed

        except Exception as error:
            last_error = error

            print(
                f"API attempt {attempt}/{MAX_RETRIES} failed: {error}"
            )

            if attempt < MAX_RETRIES:
                print(
                    f"Retrying in {RETRY_SECONDS} seconds..."
                )
                time.sleep(RETRY_SECONDS)

    raise RuntimeError(
        f"Groq request failed after {MAX_RETRIES} attempts: "
        f"{last_error}"
    )


# ============================================================
# PDF processing
# ============================================================

def extract_pages(pdf_path):
    """
    Extract selectable text page by page.
    """

    reader = PdfReader(str(pdf_path))
    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        text = page.extract_text() or ""
        text = text.strip()

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    if not pages:
        raise RuntimeError(
            "No text could be extracted from the PDF. "
            "The PDF may be scanned and require OCR."
        )

    return pages


def create_chunks(pages):
    """
    Group pages into small chunks.
    A page is never split in the middle.
    """

    chunks = []
    current_pages = []
    current_text = ""

    for page in pages:
        page_block = (
            f"[Page {page['page']}]\n"
            f"{page['text']}\n\n"
        )

        # If one page itself exceeds the limit, truncate that page.
        if len(page_block) > MAX_CHUNK_CHARS:
            page_block = page_block[:MAX_CHUNK_CHARS]

        should_flush = (
            current_text
            and len(current_text) + len(page_block)
            > MAX_CHUNK_CHARS
        )

        if should_flush:
            chunks.append(
                {
                    "pages": current_pages,
                    "text": current_text.strip(),
                }
            )

            current_pages = []
            current_text = ""

        current_pages.append(page["page"])
        current_text += page_block

    if current_text.strip():
        chunks.append(
            {
                "pages": current_pages,
                "text": current_text.strip(),
            }
        )

    return chunks


# ============================================================
# Ontology schema utilities
# ============================================================

def empty_ontology():
    """
    Return the complete ontology schema.
    """

    return {
        "terms": [],
        "evidence": [],
        "mappings": [],
        "constraints": [],
        "relations": [],
    }


def normalize_chunk_ontology(ontology):
    """
    Guarantee that every chunk has every expected list.
    """

    if not isinstance(ontology, dict):
        ontology = {}

    normalized = empty_ontology()

    for key in normalized:
        value = ontology.get(key, [])

        if isinstance(value, list):
            normalized[key] = value

    return normalized


def normalize_item_ids(items, prefix):
    """
    Replace model-generated IDs with deterministic IDs.
    """

    normalized = []

    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue

        item = dict(item)
        item["id"] = f"{prefix}_{index:04d}"
        normalized.append(item)

    return normalized


# ============================================================
# Ontology extraction
# ============================================================

def build_chunk_ontology(
    client,
    chunk,
    chunk_number,
    total_chunks,
):
    """
    Generate ontology records for one PDF chunk.
    """

    system_prompt = """
You are an evidence-grounded ontology extraction system.

Extract only concepts explicitly supported by the supplied PDF section.
Do not invent facts, names, metrics, relationships, or page numbers.

Return valid JSON only.

The JSON object must contain exactly these top-level keys:
terms, evidence, mappings, constraints, relations.

Use this schema:

{
  "terms": [
    {
      "id": "temporary_term_id",
      "name": "short canonical name",
      "description": "grounded description",
      "type": "concept|method|dataset|metric|entity|finding|limitation",
      "evidence_pages":[1]
    }
  ],
  "evidence": [
    {
      "id": "temporary_evidence_id",
      "claim": "supported claim",
      "pages":,[1]
      "quote": "short exact quote or faithful short excerpt"
    }
  ],
  "mappings": [
    {
      "id": "temporary_mapping_id",
      "term_name": "name of a term above",
      "source": "pdf",
      "location": "page 1",
      "explanation": "why the term is grounded"
    }
  ],
  "constraints": [
    {
      "id": "temporary_constraint_id",
      "statement": "constraint or condition supported by the PDF",
      "evidence_pages":[1]
    }
  ],
  "relations": [
    {
      "source_term_name": "term name",
      "relation": "supports|uses|evaluates|improves|contains|contrasts_with",
      "target_term_name": "term name",
      "evidence_pages":[1]
    }
  ]
}

Rules:
- Empty arrays are allowed.
- Keep the ontology small and precise.
- Use only page numbers shown in the supplied text.
- Every important claim must have evidence pages.
- Never use evidence from outside the supplied PDF section.
- Return JSON only.
"""

    user_prompt = f"""
Extract an ontology from chunk {chunk_number} of {total_chunks}.

This chunk contains pages:
{chunk["pages"]}

PDF text:
{chunk["text"]}
"""

    result = call_groq(
        client=client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    return normalize_chunk_ontology(result)


# ============================================================
# Ontology merging
# ============================================================

def merge_ontologies(chunk_ontologies):
    """
    Merge chunk-level ontologies into one ontology.
    """

    merged = empty_ontology()

    for chunk_number, raw_ontology in enumerate(
        chunk_ontologies,
        start=1,
    ):
        ontology = normalize_chunk_ontology(raw_ontology)

        merged["terms"].extend(
            normalize_item_ids(
                ontology["terms"],
                f"term_c{chunk_number}",
            )
        )

        merged["evidence"].extend(
            normalize_item_ids(
                ontology["evidence"],
                f"evidence_c{chunk_number}",
            )
        )

        merged["mappings"].extend(
            normalize_item_ids(
                ontology["mappings"],
                f"mapping_c{chunk_number}",
            )
        )

        merged["constraints"].extend(
            normalize_item_ids(
                ontology["constraints"],
                f"constraint_c{chunk_number}",
            )
        )

        # Relations do not necessarily have IDs.
        merged["relations"].extend(
            ontology["relations"]
        )

    return merged


# ============================================================
# Deterministic local deduplication
# ============================================================

def deduplicate_ontology(ontology):
    """
    Deduplicate locally without another Groq request.

    This avoids:
    - another TPM-consuming request
    - incomplete model-generated top-level keys
    - accidental loss of extracted information
    """

    source = normalize_chunk_ontology(ontology)
    result = empty_ontology()

    seen_terms = set()
    seen_evidence = set()
    seen_mappings = set()
    seen_constraints = set()
    seen_relations = set()

    for term in source["terms"]:
        name = str(term.get("name", "")).strip().lower()
        description = str(
            term.get("description", "")
        ).strip().lower()

        key = (name, description)

        if name and key not in seen_terms:
            seen_terms.add(key)
            result["terms"].append(term)

    for evidence in source["evidence"]:
        claim = str(
            evidence.get("claim", "")
        ).strip().lower()

        pages = tuple(
            evidence.get("pages", [])
        )

        quote = str(
            evidence.get("quote", "")
        ).strip().lower()

        key = (claim, pages, quote)

        if claim and key not in seen_evidence:
            seen_evidence.add(key)
            result["evidence"].append(evidence)

    for mapping in source["mappings"]:
        term_name = str(
            mapping.get("term_name", "")
        ).strip().lower()

        location = str(
            mapping.get("location", "")
        ).strip().lower()

        key = (term_name, location)

        if key not in seen_mappings:
            seen_mappings.add(key)
            result["mappings"].append(mapping)

    for constraint in source["constraints"]:
        statement = str(
            constraint.get("statement", "")
        ).strip().lower()

        if statement and statement not in seen_constraints:
            seen_constraints.add(statement)
            result["constraints"].append(constraint)

    for relation in source["relations"]:
        source_name = str(
            relation.get("source_term_name", "")
        ).strip().lower()

        relation_name = str(
            relation.get("relation", "")
        ).strip().lower()

        target_name = str(
            relation.get("target_term_name", "")
        ).strip().lower()

        pages = tuple(
            relation.get("evidence_pages", [])
        )

        key = (
            source_name,
            relation_name,
            target_name,
            pages,
        )

        if key not in seen_relations:
            seen_relations.add(key)
            result["relations"].append(relation)

    return result


# ============================================================
# Ontology validation
# ============================================================

def validate_ontology(ontology):
    """
    Validate the final ontology structure.
    """

    required_keys = {
        "terms",
        "evidence",
        "mappings",
        "constraints",
        "relations",
    }

    missing = required_keys.difference(ontology.keys())

    if missing:
        raise ValueError(
            "Ontology is missing top-level keys: "
            f"{sorted(missing)}"
        )

    for key in required_keys:
        if not isinstance(ontology[key], list):
            raise ValueError(
                f"Ontology field '{key}' must be a list."
            )

    for term in ontology["terms"]:
        required = {
            "id",
            "name",
            "description",
            "type",
        }

        missing_fields = required.difference(term)

        if missing_fields:
            raise ValueError(
                "Invalid term; missing fields: "
                f"{sorted(missing_fields)}"
            )

    for evidence in ontology["evidence"]:
        required = {
            "id",
            "claim",
            "pages",
            "quote",
        }

        missing_fields = required.difference(evidence)

        if missing_fields:
            raise ValueError(
                "Invalid evidence; missing fields: "
                f"{sorted(missing_fields)}"
            )

    for mapping in ontology["mappings"]:
        required = {
            "id",
            "term_name",
            "source",
            "location",
            "explanation",
        }

        missing_fields = required.difference(mapping)

        if missing_fields:
            raise ValueError(
                "Invalid mapping; missing fields: "
                f"{sorted(missing_fields)}"
            )

    for constraint in ontology["constraints"]:
        required = {
            "id",
            "statement",
            "evidence_pages",
        }

        missing_fields = required.difference(constraint)

        if missing_fields:
            raise ValueError(
                "Invalid constraint; missing fields: "
                f"{sorted(missing_fields)}"
            )

    return True


# ============================================================
# Main execution
# ============================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Reading PDF: {PDF_PATH}")

    pages = extract_pages(PDF_PATH)
    chunks = create_chunks(pages)

    print(f"Extracted pages: {len(pages)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Using model: {MODEL}")

    client = Groq(api_key=API_KEY)
    chunk_ontologies = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        print(
            f"Generating ontology for chunk "
            f"{index}/{len(chunks)}; "
            f"pages={chunk['pages']}"
        )

        ontology = build_chunk_ontology(
            client=client,
            chunk=chunk,
            chunk_number=index,
            total_chunks=len(chunks),
        )

        chunk_ontologies.append(ontology)

        if index < len(chunks):
            time.sleep(BETWEEN_CHUNKS_SECONDS)

    merged = merge_ontologies(chunk_ontologies)
    final_ontology = deduplicate_ontology(merged)

    validate_ontology(final_ontology)

    final_ontology["metadata"] = {
        "ontology_version": "v0",
        "source_type": "single_pdf",
        "generation_method": (
            "chunked_groq_extraction_and_local_deduplication"
        ),
        "model": MODEL,
        "pdf": str(PDF_PATH),
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "max_chunk_chars": MAX_CHUNK_CHARS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }

    run_data = {
        "pdf": str(PDF_PATH),
        "model": MODEL,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "chunks": [
            {
                "chunk_number": index,
                "pages": chunk["pages"],
                "character_count": len(chunk["text"]),
            }
            for index, chunk in enumerate(
                chunks,
                start=1,
            )
        ],
        "ontology": final_ontology,
    }

    ONTOLOGY_PATH.write_text(
        json.dumps(
            final_ontology,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    RUN_PATH.write_text(
        json.dumps(
            run_data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nOntology generation completed.")
    print(f"Terms: {len(final_ontology['terms'])}")
    print(f"Evidence: {len(final_ontology['evidence'])}")
    print(f"Mappings: {len(final_ontology['mappings'])}")
    print(f"Constraints: {len(final_ontology['constraints'])}")
    print(f"Relations: {len(final_ontology['relations'])}")
    print(f"\nSaved ontology: {ONTOLOGY_PATH}")
    print(f"Saved run metadata: {RUN_PATH}")


if __name__ == "__main__":
    main()