from pathlib import Path
import re
import polars as pl

PROJECT_ROOT = Path(r"C:/Users/Rahul/OneDrive/Desktop/Learning/Projects/nyc_taxi")
RAW_ROOT = PROJECT_ROOT / "raw"
OUT_DIR = PROJECT_ROOT / "interim"

MONEY_COLUMNS = [
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "airport_fee",
    "cbd_congestion_fee",
]

STANDARD_COLUMNS = [
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
    "cbd_congestion_fee",
]

def discover_files(raw_root: Path) -> list[Path]:
    return sorted(raw_root.rglob("*.parquet"))

def infer_year_month(path: Path) -> tuple[int, int]:
    match = re.search(r"yellow_tripdata_(\d{4})-(\d{2})\.parquet", str(path))
    if not match:
        raise ValueError(f"Could not infer year/month from path: {path}")
    return int(match.group(1)), int(match.group(2))

def normalize_column_names(lf: pl.LazyFrame) -> pl.LazyFrame:
    schema_names = lf.collect_schema().names()

    rename_map = {}
    if "Airport_fee" in schema_names:
        rename_map["Airport_fee"] = "airport_fee"

    if rename_map:
        lf = lf.rename(rename_map)

    return lf

def add_missing_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    existing = set(lf.collect_schema().names())
    missing = [c for c in STANDARD_COLUMNS if c not in existing]

    if missing:
        lf = lf.with_columns([pl.lit(None).alias(c) for c in missing])

    return lf

def cast_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    exprs = [
        pl.col("VendorID").cast(pl.Int64, strict=False),
        pl.col("tpep_pickup_datetime").cast(pl.Datetime, strict=False),
        pl.col("tpep_dropoff_datetime").cast(pl.Datetime, strict=False),
        pl.col("passenger_count").cast(pl.Float64, strict=False),
        pl.col("trip_distance").cast(pl.Float64, strict=False),
        pl.col("RatecodeID").cast(pl.Float64, strict=False),
        pl.col("store_and_fwd_flag").cast(pl.String, strict=False),
        pl.col("PULocationID").cast(pl.Int64, strict=False),
        pl.col("DOLocationID").cast(pl.Int64, strict=False),
        pl.col("payment_type").cast(pl.Int64, strict=False),
    ]

    exprs += [pl.col(c).cast(pl.Float64, strict=False) for c in MONEY_COLUMNS]

    return lf.with_columns(exprs)

def normalize_file(path: Path) -> pl.LazyFrame:
    year, month = infer_year_month(path)

    lf = pl.scan_parquet(str(path))
    lf = normalize_column_names(lf)
    lf = add_missing_columns(lf)
    lf = cast_columns(lf)

    lf = lf.select(STANDARD_COLUMNS).with_columns(
        [
            pl.lit(year).alias("source_year"),
            pl.lit(month).alias("source_month"),
            pl.lit(path.name).alias("source_file"),
        ]
    )

    return lf

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = discover_files(RAW_ROOT)
    print(f"Found {len(files)} parquet files.")

    yearly_lfs: dict[int, list[pl.LazyFrame]] = {}

    for path in files:
        year, month = infer_year_month(path)
        print(f"Normalizing {path.name}")
        yearly_lfs.setdefault(year, []).append(normalize_file(path))

    for year, lfs in yearly_lfs.items():
        print(f"\nCombining year {year} ({len(lfs)} files)...")
        yearly_df = pl.concat(lfs, how="vertical_relaxed").collect(streaming=True)

        out_path = OUT_DIR / f"yellow_taxi_{year}_normalized.parquet"
        yearly_df.write_parquet(out_path)

        print(f"Wrote: {out_path}")
        print(f"Shape: {yearly_df.shape}")

if __name__ == "__main__":
    main()