import json
import csv
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("data/fever")

FEVER_TRAIN = BASE_DIR / "train.jsonl"
WIKI_DIR = BASE_DIR / "wiki-pages"

CLAIM_OUTPUT = BASE_DIR / "fever_claims_1000.csv"
EVIDENCE_OUTPUT = BASE_DIR / "fever_evidence_1000.csv"

MAX_CLAIMS = 1000


# ============================================================
# 1. LOAD FEVER CLAIMS
# ============================================================

def load_claims():

    claims = []

    print("=" * 70)
    print("STEP 1 - LOADING FEVER CLAIMS")
    print("=" * 70)

    with open(
        FEVER_TRAIN,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            try:
                item = json.loads(line)

            except json.JSONDecodeError:
                continue

            claims.append(item)

            if len(claims) >= MAX_CLAIMS:
                break

    print(f"Claims loaded: {len(claims)}")

    return claims


# ============================================================
# 2. BUILD WIKIPEDIA PAGE INDEX
# ============================================================

def build_wiki_index():

    print()
    print("=" * 70)
    print("STEP 2 - INDEXING WIKIPEDIA PAGES")
    print("=" * 70)

    wiki_files = sorted(
        WIKI_DIR.glob("wiki-*.jsonl")
    )

    print(
        f"Wikipedia files found: "
        f"{len(wiki_files)}"
    )

    page_to_file = {}

    for i, file_path in enumerate(
        wiki_files,
        start=1
    ):

        print(
            f"[{i}/{len(wiki_files)}] "
            f"{file_path.name}"
        )

        try:

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as f:

                for line in f:

                    if not line.strip():
                        continue

                    try:
                        page = json.loads(line)

                    except json.JSONDecodeError:
                        continue

                    page_id = page.get("id")

                    if page_id is not None:

                        page_to_file[
                            str(page_id)
                        ] = file_path

        except Exception as e:

            print(
                f"Warning: {file_path.name}: {e}"
            )

    print()
    print(
        f"Indexed Wikipedia pages: "
        f"{len(page_to_file)}"
    )

    return page_to_file


# ============================================================
# 3. LOAD WIKIPEDIA PAGE
# ============================================================

def get_page(
    page_id,
    page_to_file,
    page_cache
):

    page_id = str(page_id)

    # --------------------------------------------------------
    # Check cache first
    # --------------------------------------------------------

    if page_id in page_cache:

        return page_cache[page_id]

    file_path = page_to_file.get(page_id)

    if file_path is None:

        return None

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                if not line.strip():
                    continue

                try:
                    page = json.loads(line)

                except json.JSONDecodeError:
                    continue

                if str(page.get("id")) == page_id:

                    page_cache[page_id] = page

                    return page

    except Exception:

        return None

    return None


# ============================================================
# 4. EXTRACT SENTENCE
# ============================================================

def extract_sentence(
    page,
    sentence_id
):

    if page is None:
        return None

    if sentence_id is None:
        return None

    try:

        sentence_id = int(sentence_id)

    except (
        TypeError,
        ValueError
    ):

        return None

    if sentence_id < 0:
        return None

    lines_data = page.get("lines")

    if lines_data is None:
        return None

    # FEVER normally stores lines as one string
    if isinstance(lines_data, str):

        lines = lines_data.split("\n")

    elif isinstance(lines_data, list):

        lines = lines_data

    else:

        return None

    if sentence_id >= len(lines):

        return None

    raw_line = lines[sentence_id]

    if not raw_line:
        return None

    parts = raw_line.split("\t")

    if len(parts) >= 2:

        sentence = parts[1].strip()

    else:

        sentence = raw_line.strip()

    if not sentence:
        return None

    return sentence


# ============================================================
# 5. PROCESS CLAIMS
# ============================================================

def process_claims(
    claims,
    page_to_file
):

    claim_rows = []
    evidence_rows = []

    page_cache = {}

    missing_pages = 0
    missing_sentences = 0
    malformed_items = 0

    # Used to prevent duplicate evidence rows
    seen_evidence = set()

    print()
    print("=" * 70)
    print("STEP 3 - EXTRACTING CLAIMS AND EVIDENCE")
    print("=" * 70)

    for claim_number, item in enumerate(
        claims,
        start=1
    ):

        claim_id = item.get("id")
        claim = item.get("claim")
        label = item.get("label")

        evidence_groups = item.get(
            "evidence",
            []
        )

        claim_evidence = []

        # ----------------------------------------------------
        # Process evidence groups
        # ----------------------------------------------------

        if isinstance(
            evidence_groups,
            list
        ):

            for group_number, group in enumerate(
                evidence_groups
            ):

                if not isinstance(
                    group,
                    list
                ):

                    malformed_items += 1
                    continue

                for evidence_item in group:

                    if not isinstance(
                        evidence_item,
                        list
                    ):

                        malformed_items += 1
                        continue

                    if len(evidence_item) < 4:

                        malformed_items += 1
                        continue

                    page_id = evidence_item[2]
                    sentence_id = evidence_item[3]

                    # ----------------------------------------
                    # Missing page
                    # ----------------------------------------

                    if page_id is None:

                        missing_pages += 1
                        continue

                    # ----------------------------------------
                    # Missing sentence
                    # ----------------------------------------

                    if sentence_id is None:

                        missing_sentences += 1
                        continue

                    try:

                        sentence_id = int(
                            sentence_id
                        )

                    except (
                        TypeError,
                        ValueError
                    ):

                        missing_sentences += 1
                        continue

                    # ----------------------------------------
                    # Load page
                    # ----------------------------------------

                    page = get_page(
                        page_id,
                        page_to_file,
                        page_cache
                    )

                    if page is None:

                        missing_pages += 1
                        continue

                    # ----------------------------------------
                    # Extract sentence
                    # ----------------------------------------

                    evidence_sentence = extract_sentence(
                        page,
                        sentence_id
                    )

                    if evidence_sentence is None:

                        missing_sentences += 1
                        continue

                    # ----------------------------------------
                    # Duplicate protection
                    # ----------------------------------------

                    evidence_key = (
                        str(claim_id),
                        str(page_id),
                        int(sentence_id)
                    )

                    if evidence_key in seen_evidence:

                        continue

                    seen_evidence.add(
                        evidence_key
                    )

                    # ----------------------------------------
                    # Evidence row
                    # ----------------------------------------

                    evidence_rows.append({

                        "claim_id": claim_id,

                        "claim": claim,

                        "label": label,

                        "evidence_group":
                            group_number,

                        "page_id": page_id,

                        "sentence_id":
                            sentence_id,

                        "evidence_sentence":
                            evidence_sentence

                    })

                    claim_evidence.append(
                        evidence_sentence
                    )

        # ----------------------------------------------------
        # Claim-level row
        # ----------------------------------------------------

        unique_evidence = list(
            dict.fromkeys(
                claim_evidence
            )
        )

        claim_rows.append({

            "claim_id": claim_id,

            "claim": claim,

            "label": label,

            "evidence_available":
                1 if unique_evidence else 0,

            "num_evidence_sentences":
                len(unique_evidence),

            "evidence_sentences":
                " || ".join(
                    unique_evidence
                )

        })

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if claim_number % 100 == 0:

            print(
                f"Processed: "
                f"{claim_number}/{len(claims)} "
                f"| Evidence rows: "
                f"{len(evidence_rows)}"
            )

    print()
    print(
        f"Missing pages: {missing_pages}"
    )

    print(
        f"Missing sentences: "
        f"{missing_sentences}"
    )

    print(
        f"Malformed evidence items: "
        f"{malformed_items}"
    )

    return (
        claim_rows,
        evidence_rows
    )


# ============================================================
# 6. SAVE CLAIM-LEVEL DATASET
# ============================================================

def save_claim_dataset(rows):

    print()
    print("=" * 70)
    print("STEP 4 - SAVING CLAIM-LEVEL DATASET")
    print("=" * 70)

    fields = [

        "claim_id",

        "claim",

        "label",

        "evidence_available",

        "num_evidence_sentences",

        "evidence_sentences"

    ]

    with open(
        CLAIM_OUTPUT,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(rows)

    print(
        f"Saved: {CLAIM_OUTPUT}"
    )

    print(
        f"Rows: {len(rows)}"
    )


# ============================================================
# 7. SAVE EVIDENCE-LEVEL DATASET
# ============================================================

def save_evidence_dataset(rows):

    print()
    print("=" * 70)
    print("STEP 5 - SAVING EVIDENCE-LEVEL DATASET")
    print("=" * 70)

    fields = [

        "claim_id",

        "claim",

        "label",

        "evidence_group",

        "page_id",

        "sentence_id",

        "evidence_sentence"

    ]

    with open(
        EVIDENCE_OUTPUT,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(rows)

    print(
        f"Saved: {EVIDENCE_OUTPUT}"
    )

    print(
        f"Rows: {len(rows)}"
    )


# ============================================================
# 8. PRINT DATASET STATISTICS
# ============================================================

def print_statistics(
    claim_rows,
    evidence_rows
):

    print()
    print("=" * 70)
    print("FINAL DATASET STATISTICS")
    print("=" * 70)

    print(
        f"Total claims: "
        f"{len(claim_rows)}"
    )

    print(
        f"Total unique evidence rows: "
        f"{len(evidence_rows)}"
    )

    # --------------------------------------------------------
    # Label distribution
    # --------------------------------------------------------

    labels = {}

    for row in claim_rows:

        label = row["label"]

        labels[label] = (
            labels.get(label, 0) + 1
        )

    print()
    print("Claim label distribution:")

    for label, count in labels.items():

        print(
            f"  {label}: {count}"
        )

    # --------------------------------------------------------
    # Evidence availability
    # --------------------------------------------------------

    evidence_yes = sum(
        1
        for row in claim_rows
        if row["evidence_available"] == 1
    )

    evidence_no = (
        len(claim_rows) - evidence_yes
    )

    print()
    print(
        f"Claims with evidence: "
        f"{evidence_yes}"
    )

    print(
        f"Claims without evidence: "
        f"{evidence_no}"
    )

    print()
    print("=" * 70)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    # Check files
    if not FEVER_TRAIN.exists():

        print(
            f"ERROR: FEVER train file not found:\n"
            f"{FEVER_TRAIN}"
        )

        return

    if not WIKI_DIR.exists():

        print(
            f"ERROR: Wikipedia directory not found:\n"
            f"{WIKI_DIR}"
        )

        return

    # Step 1
    claims = load_claims()

    # Step 2
    page_to_file = build_wiki_index()

    # Step 3
    claim_rows, evidence_rows = process_claims(
        claims,
        page_to_file
    )

    # Step 4
    save_claim_dataset(
        claim_rows
    )

    # Step 5
    save_evidence_dataset(
        evidence_rows
    )

    # Step 6
    print_statistics(
        claim_rows,
        evidence_rows
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()