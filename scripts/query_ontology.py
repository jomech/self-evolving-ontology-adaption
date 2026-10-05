import json
import re
import sys
from pathlib import Path


ONTOLOGY_PATH = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "outputs/ontology_v0.json"
)

QUERY = (
    sys.argv[2]
    if len(sys.argv) > 2
    else "method evaluation results limitations"
)


def tokenize(text):
    return set(
        re.findall(
            r"[a-zA-Z0-9_]+",
            text.lower(),
        )
    )


def searchable_text(record):
    return json.dumps(
        record,
        ensure_ascii=False,
    ).lower()


def browse_semantics(
    ontology,
    query,
    kind="all",
    limit=10,
):
    query_tokens = tokenize(query)
    results = []

    groups = [
        "terms",
        "evidence",
        "mappings",
        "constraints",
        "relations",
    ]

    if kind != "all":
        groups = [kind]

    for group in groups:
        for record in ontology.get(group, []):
            text = searchable_text(record)
            record_tokens = tokenize(text)

            overlap = query_tokens.intersection(
                record_tokens
            )

            if overlap:
                score = len(overlap)

                results.append(
                    {
                        "kind": group,
                        "score": score,
                        "record": record,
                    }
                )

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:limit]


def resolve_semantics(ontology, ids):
    resolved = []

    for group, records in ontology.items():
        if not isinstance(records, list):
            continue

        for record in records:
            if record.get("id") in ids:
                resolved.append(
                    {
                        "kind": group,
                        "record": record,
                    }
                )

    return resolved


def main():
    if not ONTOLOGY_PATH.exists():
        raise FileNotFoundError(
            f"Ontology not found: {ONTOLOGY_PATH}"
        )

    ontology = json.loads(
        ONTOLOGY_PATH.read_text(
            encoding="utf-8"
        )
    )

    results = browse_semantics(
        ontology=ontology,
        query=QUERY,
        kind="all",
        limit=10,
    )

    print(f"\nQuery: {QUERY}")
    print(f"Ontology: {ONTOLOGY_PATH}")
    print(f"Matches: {len(results)}\n")

    for index, item in enumerate(
        results,
        start=1,
    ):
        print("=" * 70)
        print(f"Result {index}")
        print(f"Kind: {item['kind']}")
        print(f"Score: {item['score']}")
        print(
            json.dumps(
                item["record"],
                indent=2,
                ensure_ascii=False,
            )
        )

    ids = [
        item["record"]["id"]
        for item in results
        if "id" in item["record"]
    ]

    resolved = resolve_semantics(
        ontology=ontology,
        ids=ids,
    )

    Path("outputs").mkdir(exist_ok=True)

    output = {
        "query": QUERY,
        "browse_results": results,
        "resolved_records": resolved,
    }

    Path(
        "outputs/ontology_query_result.json"
    ).write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSaved:")
    print("outputs/ontology_query_result.json")


if __name__ == "__main__":
    main()