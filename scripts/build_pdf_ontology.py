import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

load_dotenv()

pdf_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/company_report.pdf")
model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing from .env")

reader = PdfReader(str(pdf_path))
page_text = []

for number, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""
    if text.strip():
        page_text.append({"page": number, "text": text})

document = "\n\n".join(
    f"[Page {item['page']}]\n{item['text']}" for item in page_text
)

document = document[:100_000]

prompt = f"""Build an evidence-grounded ontology for this PDF.

Return valid JSON only with this structure:

{{
  "terms": [
    {{
      "id": "term_001",
      "name": "...",
      "description": "...",
      "type": "concept|method|dataset|metric|entity|finding",
      "evidence_pages": [1]
    }}
  ],
  "mappings": [
    {{
      "id": "mapping_001",
      "term_id": "term_001",
      "source": "pdf",
      "location": "page 1",
      "explanation": "..."
    }}
  ],
  "constraints": [
    {{
      "id": "constraint_001",
      "statement": "...",
      "evidence_pages": [1]
    }}
  ],
  "evidence": [
    {{
      "id": "evidence_001",
      "claim": "...",
      "pages": [1],
      "quote": "short exact quote or faithful excerpt"
    }}
  ],
  "relations": [
    {{
      "source_term_id": "term_001",
      "relation": "supports|uses|evaluates|improves|contrasts_with|contains",
      "target_term_id": "term_002"
    }}
  ]
}}

Rules:
- Use only concepts supported by the PDF.
- Every term must have evidence_pages.
- Every important mapping must identify a page.
- Do not invent metrics or results.
- Keep evidence quotes short.
- Return JSON only.

PDF:
{document}
"""

client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model=model,
    temperature=0,
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "system",
            "content": "You create small, evidence-grounded semantic graphs.",
        },
        {"role": "user", "content": prompt},
    ],
)

ontology = json.loads(response.choices[0].message.content)

Path("outputs").mkdir(exist_ok=True)
Path("outputs/ontology_v0.json").write_text(
    json.dumps(ontology, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("Created outputs/ontology_v0.json")
print("Terms:", len(ontology.get("terms", [])))
print("Mappings:", len(ontology.get("mappings", [])))
print("Constraints:", len(ontology.get("constraints", [])))
print("Evidence:", len(ontology.get("evidence", [])))
print("Relations:", len(ontology.get("relations", [])))