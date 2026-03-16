from pathlib import Path
import re
import polars as pl

# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT = Path(r"C:/Users/Rahul/OneDrive/Desktop/Learning/Projects/nyc_taxi")
RAW_ROOT = PROJECT_ROOT / "raw"
REPORT_DIR = PROJECT_ROOT / "reports" / "data_quality"
OUT_CSV = REPORT_DIR / "schema_audit.csv"

# --------------------------------------------------
# Expected columns for yellow taxi fare modeling from the dataset
# --------------------------------------------------
CANONICAL_COLUMNS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "airport_fee",
]

def discover_files(raw_root: Path) -> list[Path]:
    return sorted(raw_root.rglob("*.parquet"))

def infer_year_month(path: Path) -> tuple[str | None, str | None]:
    path_str = str(path)
    match = re.search(r"yellow_tripdata_(\d{4})-(\d{2})\.parquet", path_str)
    if match:
        return match.group(1), match.group(2)
    return None, None

def audit_file(path: Path) -> dict:
    lf = pl.scan_parquet(str(path))
    schema = lf.collect_schema()

    columns = list(schema.names())
    dtypes = {col: str(dtype) for col, dtype in schema.items()}

    missing_canonical = [c for c in CANONICAL_COLUMNS if c not in columns]
    extra_columns = [c for c in columns if c not in CANONICAL_COLUMNS]

    year, month = infer_year_month(path)

    return {
        "file_path": str(path),
        "year": year,
        "month": month,
        "n_columns": len(columns),
        "columns": " | ".join(columns),
        "missing_canonical": " | ".join(missing_canonical),
        "extra_columns": " | ".join(extra_columns),
        "dtype_map": str(dtypes),
    }

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    files = discover_files(RAW_ROOT)
    print(f"Found {len(files)} parquet files.")

    if not files:
        raise FileNotFoundError(f"No parquet files found under {RAW_ROOT}")

    audit_rows = []
    for i, path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Auditing: {path.name}")
        audit_rows.append(audit_file(path))

    df_audit = pl.DataFrame(audit_rows)
    df_audit.write_csv(OUT_CSV)

    print(f"\nWrote schema audit to:\n{OUT_CSV}")

    # Quick summaries
    print("\n=== Missing canonical columns summary ===")
    missing_summary = (
        df_audit
        .filter(pl.col("missing_canonical") != "")
        .select(["year", "month", "file_path", "missing_canonical"])
    )
    print(missing_summary if missing_summary.height > 0 else "No missing canonical columns found.")

    print("\n=== Extra columns summary ===")
    extra_summary = (
        df_audit
        .filter(pl.col("extra_columns") != "")
        .select(["year", "month", "file_path", "extra_columns"])
    )
    print(extra_summary if extra_summary.height > 0 else "No extra columns found.")

if __name__ == "__main__":
    main()