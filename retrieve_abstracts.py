

CSV_SEPARATOR = ";"
DOI_COLUMN    = "DOI"
EXPECTED_COLS = {"Title", "DOI", "Decision"}

import re
import time
import argparse
import logging
import requests
import pandas as pd
from tqdm import tqdm
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


DOI_URL_RE = re.compile(r"https?://(?:dx\.)?doi\.org/(.+)", re.IGNORECASE)


def normalise_doi(raw: str) -> str | None:
    if not isinstance(raw, str):
        return None
    raw = raw.strip()
    m = DOI_URL_RE.match(raw)
    if m:
        return m.group(1).strip()
    if raw.startswith("10."):
        return raw
    return None



def _get(url: str, params: dict = None, timeout: int = 10) -> dict | None:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def fetch_semantic_scholar(doi: str, **_) -> str | None:
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    data = _get(url, params={"fields": "abstract"})
    if data:
        return data.get("abstract") or None
    return None


def fetch_crossref(doi: str, email: str = None, **_) -> str | None:
    """
    Crossref REST API.
    Providing an email activates the polite pool (higher rate limits).
    """
    url = f"https://api.crossref.org/works/{doi}"
    params = {}
    if email:
        params["mailto"] = email
    data = _get(url, params=params)
    if data:
        abstract = data.get("message", {}).get("abstract", "")
        if abstract:
            abstract = re.sub(r"<[^>]+>", " ", abstract).strip()
            abstract = re.sub(r"\s+", " ", abstract)
            return abstract or None
    return None


def fetch_pubmed(doi: str, email: str = None, **_) -> str | None:
    """
    NCBI Entrez — two-step lookup:
      1. esearch  : DOI  → PubMed ID
      2. efetch   : PMID → abstract XML
    """
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    search_params = {
        "db": "pubmed",
        "term": f"{doi}[doi]",
        "retmode": "json",
        "retmax": 1,
    }
    if email:
        search_params["email"] = email

    search = _get(f"{base}/esearch.fcgi", params=search_params)
    if not search:
        return None
    ids = search.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return None

    fetch_params = {
        "db": "pubmed",
        "id": ids[0],
        "rettype": "abstract",
        "retmode": "xml",
    }
    if email:
        fetch_params["email"] = email

    try:
        r = requests.get(f"{base}/efetch.fcgi", params=fetch_params, timeout=10)
        if r.status_code == 200:
            parts = re.findall(
                r"<AbstractText[^>]*>(.*?)</AbstractText>", r.text, re.DOTALL
            )
            if parts:
                abstract = " ".join(
                    re.sub(r"<[^>]+>", " ", p).strip() for p in parts
                )
                return re.sub(r"\s+", " ", abstract).strip() or None
    except Exception:
        pass
    return None


def fetch_openalex(doi: str, **_) -> str | None:
    url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    data = _get(url, params={"select": "abstract_inverted_index"})
    if not data:
        return None

    inv_index = data.get("abstract_inverted_index")
    if not inv_index:
        return None

    pairs = []
    for word, positions in inv_index.items():
        for pos in positions:
            pairs.append((pos, word))
    pairs.sort()
    abstract = " ".join(w for _, w in pairs)
    return abstract.strip() or None


FETCHERS = [
    ("Semantic Scholar", fetch_semantic_scholar),
    ("Crossref",         fetch_crossref),
    ("PubMed",           fetch_pubmed),
    ("OpenAlex",         fetch_openalex),
]


def fetch_abstract(
    raw_doi: str,
    email: str = None,
    delay: float = 0.5,
) -> tuple[str | None, str | None]:

    doi = normalise_doi(raw_doi)
    if doi is None:
        log.debug(f"  Skipping unparseable DOI: {raw_doi!r}")
        return None, "invalid_doi"

    for name, fetcher in FETCHERS:
        try:
            abstract = fetcher(doi, email=email)
        except Exception as exc:
            log.debug(f"  [{name}] exception for {doi}: {exc}")
            abstract = None

        if abstract:
            return abstract, name

        time.sleep(delay)   

    return None, None



def process_file(
    input_path: str,
    output_path: str = None,
    email: str = None,
    delay: float = 0.5,
):
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Reading: {input_path}")
    df = pd.read_csv(input_path, sep=CSV_SEPARATOR, dtype=str)
    log.info(f"Loaded {len(df)} rows | Columns: {list(df.columns)}")

    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    abstracts, sources = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching abstracts"):
        abstract, source = fetch_abstract(row[DOI_COLUMN], email=email, delay=delay)
        abstracts.append(abstract)
        sources.append(source)

    df["Abstract"]        = abstracts
    df["abstract_source"] = sources

    df.to_csv(output_path, sep=CSV_SEPARATOR, index=False)

    found = sum(1 for a in abstracts if a is not None)
    log.info(f"Done. Saved → {output_path}")
    log.info(
        f"Coverage: {found}/{len(df)} "
        f"({100 * found / len(df):.1f}% retrieved)"
    )

    source_counts = pd.Series(sources).value_counts()
    log.info("Breakdown by source:\n" + source_counts.to_string())

def main():
    parser = argparse.ArgumentParser(
        description="Fetch abstracts for AINA dataset papers via DOI."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to input CSV (semicolon-separated, must contain a DOI column).",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output CSV path. Defaults to overwriting the input file in place.",
    )
    parser.add_argument(
        "--email", "-e", default=None,
        help="Your email — enables Crossref polite pool & improves PubMed limits.",
    )
    parser.add_argument(
        "--delay", "-d", type=float, default=0.5,
        help="Seconds to sleep between API calls per paper (default 0.5).",
    )
    args = parser.parse_args()
    process_file(args.input, args.output, args.email, args.delay)


if __name__ == "__main__":
    main()