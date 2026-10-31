import json
from pathlib import Path


FEVER_DIR = Path("data/fever")
WIKI_DIR = FEVER_DIR / "wiki-pages"


def load_claims(file_path):
    """Load FEVER claims from a JSONL file."""
    claims = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                claims.append(json.loads(line))

    return claims


def build_wiki_index():
    """
    Build an index:
        page title -> wiki file

    We do not load the complete Wikipedia corpus into memory.
    """

    index = {}

    wiki_files = sorted(WIKI_DIR.glob("wiki-*.jsonl"))

    print(f"Found {len(wiki_files)} Wikipedia files.")

    for file_path in wiki_files:

        print(f"Indexing: {file_path.name}")

        with open(file_path, "r", encoding="utf-8") as f:

            for line in f:

                if not line.strip():
                    continue

                page = json.loads(line)

                page_id = page.get("id")

                if page_id is not None:
                    index[str(page_id)] = file_path

    return index


def find_page(page_id, wiki_index):
    """Find the Wikipedia file containing a page."""

    return wiki_index.get(str(page_id))


def get_wiki_page(page_id, wiki_index):

    file_path = find_page(page_id, wiki_index)

    if file_path is None:
        return None

    with open(file_path, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            page = json.loads(line)

            if str(page.get("id")) == str(page_id):
                return page

    return None


def main():

    train_file = FEVER_DIR / "train.jsonl"

    print("Loading FEVER training data...")

    claims = load_claims(train_file)

    print(f"Total claims loaded: {len(claims)}")

    # Use the same sample we already inspected.
    sample_claim = claims[0]

    print("\nSample claim:")
    print(sample_claim["claim"])

    print("Label:")
    print(sample_claim["label"])

    print("\nEvidence references:")
    print(sample_claim["evidence"])

    print("\nBuilding Wikipedia index...")

    wiki_index = build_wiki_index()

    print(f"\nIndexed Wikipedia pages: {len(wiki_index)}")

    # Get first evidence group
    evidence_group = sample_claim.get("evidence", [[]])[0]

    if not evidence_group:
        print("No evidence available.")
        return

    print("\nRetrieving actual Wikipedia evidence...")

    for evidence_item in evidence_group:

        page_id = evidence_item[2]
        sentence_id = evidence_item[3]

        print("\nPage:", page_id)
        print("Sentence ID:", sentence_id)

        page = get_wiki_page(page_id, wiki_index)

        if page is None:
            print("Page not found.")
            continue

        print("Wikipedia page found.")

        sentences = page.get("lines", "").split("\n")

        if sentence_id < len(sentences):

            sentence = sentences[sentence_id]

            print("Evidence sentence:")
            print(sentence)

        else:

            print("Sentence ID not found in page.")


if __name__ == "__main__":
    main()