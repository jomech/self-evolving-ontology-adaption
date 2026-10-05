import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

load_dotenv()

pdf_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/company_report.pdf")
question = (
    sys.argv[2]
    if len(sys.argv) > 2
    else "What are the main findings, methods, limitations, and quantitative results?"
)

api_key = os.getenv("GROQ_API_KEY")
model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing from .env")

reader = PdfReader(str(pdf_path))
pages = []

for number, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""
    if text.strip():
        pages.append(f"[Page {number}]\n{text}")

document = "\n\n".join(pages)

if not document.strip():
    raise RuntimeError("No extractable text found in the PDF")

max_chars = 20_000
document = document[:max_chars]

client = Groq(api_key=api_key)

system_prompt = """You are a careful research-paper analysis agent.
Use only the supplied PDF text.
Do not invent facts.
For every important claim, include the page number when available.
If the answer is not supported by the document, say so.
Return a structured answer with:
1. Direct answer
2. Evidence
3. Uncertainty or limitations
"""

user_prompt = f"""Question:
{question}

PDF text:
{document}
"""

response = client.chat.completions.create(
    model=model,
    temperature=0,
    max_tokens=1500,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
)

answer = response.choices[0].message.content

result = {
    "pdf": str(pdf_path),
    "model": model,
    "question": question,
    "answer": answer,
    "pages": len(reader.pages),
    "extracted_characters": len(document),
}

Path("outputs").mkdir(exist_ok=True)
Path("outputs/pdf_analysis.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(answer)
print("\nSaved: outputs/pdf_analysis.json")