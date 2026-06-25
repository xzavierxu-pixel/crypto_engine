from __future__ import annotations

import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
ENV_FILE = SCRIPT_DIR / ".env"

DATA_TYPE = "poly_l2"
SLUGS_FILE = Path("artifacts/pmdata/slugs.txt")
OUTPUT_DIR = Path("artifacts/pmdata/poly_l2")
FAILED_LOG = Path("artifacts/pmdata/poly_l2_failed_downloads.csv")

MAX_WORKERS = 16
MAX_RETRIES = 3
TIMEOUT_SECONDS = 120
CHUNK_SIZE = 8 * 1024 * 1024


def load_api_key() -> str:
    key = os.environ.get("PMDATA_API_KEY") or read_env_file()
    if not key:
        raise RuntimeError(
            "Missing PMDATA_API_KEY. Set it in PowerShell or in "
            "scripts/data/step0_pmdata/.env"
        )
    return key


def read_env_file() -> str | None:
    if not ENV_FILE.exists():
        return None

    for line in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        name, sep, value = line.partition("=")
        if sep and name.strip() == "PMDATA_API_KEY":
            return value.strip().strip("\"'") or None

    return None


def load_slugs() -> list[str]:
    if not SLUGS_FILE.exists():
        raise FileNotFoundError(f"Missing slug file: {SLUGS_FILE}")

    seen: set[str] = set()
    slugs: list[str] = []

    for line in SLUGS_FILE.read_text(encoding="utf-8").splitlines():
        slug = line.strip()
        if slug and not slug.startswith("#") and slug not in seen:
            seen.add(slug)
            slugs.append(slug)

    return slugs


def safe_name(slug: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    return "".join(ch if ch in allowed else "_" for ch in slug)


def download_slug(slug: str, api_key: str) -> dict:
    output_path = OUTPUT_DIR / f"{safe_name(slug)}.parquet"
    if output_path.exists() and output_path.stat().st_size > 0:
        return {"slug": slug, "status": "skipped", "bytes": output_path.stat().st_size, "error": ""}

    url = f"https://api.pmdata.dev/download/{DATA_TYPE}/{slug}.parquet"
    headers = {"api_key": api_key, "User-Agent": "Mozilla/5.0"}
    temp_path = output_path.with_suffix(".parquet.part")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            temp_path.unlink(missing_ok=True)

            with requests.get(url, headers=headers, stream=True, timeout=TIMEOUT_SECONDS) as response:
                if response.status_code == 404:
                    return {"slug": slug, "status": "no_data", "bytes": 0, "error": "HTTP 404"}

                response.raise_for_status()

                with temp_path.open("wb") as f:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)

            temp_path.replace(output_path)
            return {"slug": slug, "status": "success", "bytes": output_path.stat().st_size, "error": ""}

        except Exception as exc:
            temp_path.unlink(missing_ok=True)
            if attempt == MAX_RETRIES:
                return {"slug": slug, "status": "failed", "bytes": 0, "error": repr(exc)}
            time.sleep(2 ** attempt)

    return {"slug": slug, "status": "failed", "bytes": 0, "error": "unknown error"}


def write_failed_log(results: list[dict]) -> None:
    failed = [row for row in results if row["status"] == "failed"]
    if not failed:
        return

    FAILED_LOG.parent.mkdir(parents=True, exist_ok=True)
    with FAILED_LOG.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["slug", "status", "bytes", "error"])
        writer.writeheader()
        writer.writerows(failed)


def main() -> None:
    api_key = load_api_key()
    slugs = load_slugs()

    print(f"Data type: {DATA_TYPE}")
    print(f"Slugs:     {len(slugs):,}")
    print(f"Output:    {OUTPUT_DIR.resolve()}")

    results: list[dict] = []
    counts = {"success": 0, "skipped": 0, "no_data": 0, "failed": 0}
    total_bytes = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_slug, slug, api_key) for slug in slugs]
        with tqdm(total=len(futures), desc=f"Downloading {DATA_TYPE}", unit="market") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                counts[result["status"]] += 1
                total_bytes += int(result["bytes"] or 0)

                pbar.update(1)
                pbar.set_postfix(counts)

    write_failed_log(results)

    print("\nFinished")
    print(f"Success:       {counts['success']:,}")
    print(f"Skipped:       {counts['skipped']:,}")
    print(f"No data:       {counts['no_data']:,}")
    print(f"Failed:        {counts['failed']:,}")
    print(f"Bytes present: {total_bytes:,}")
    if counts["failed"]:
        print(f"Failure log:   {FAILED_LOG.resolve()}")


if __name__ == "__main__":
    main()
