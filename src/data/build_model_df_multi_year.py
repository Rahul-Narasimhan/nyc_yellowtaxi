from pathlib import Path
import polars as pl

PROJECT_ROOT = Path(r"C:/Users/Rahul/OneDrive/Desktop/Learning/Projects/nyc_taxi")
INTERIM_DIR = PROJECT_ROOT / "interim"
PROCESSED_DIR = PROJECT_ROOT / "processed"

INPUT_FILES = [
    INTERIM_DIR / "yellow_taxi_2023_normalized.parquet",
    INTERIM_DIR / "yellow_taxi_2024_normalized.parquet",
    INTERIM_DIR / "yellow_taxi_2025_normalized.parquet",
]

# First-pass clean modeling thresholds
# These are not "pure structural" rules; they are deliberate modeling choices.
MIN_FARE = 0.0
MIN_TRIP_DISTANCE = 0.0
MIN_DURATION_MINUTES = 0.0


def get_valid_date_filter(year: int) -> pl.Expr:
    if year in [2023, 2024]:
        return pl.col("pickup_year") == year
    elif year == 2025:
        return (
            (pl.col("pickup_year") == 2025)
            & (pl.col("pickup_month") >= 1)
            & (pl.col("pickup_month") <= 11)
        )
    else:
        raise ValueError(f"Unexpected year parsed from file name: {year}")


def build_model_ready_lf(path: Path) -> pl.LazyFrame:
    year = int(path.stem.split("_")[2])  # yellow_taxi_2023_normalized.parquet -> 2023
    lf = pl.scan_parquet(str(path))

    lf = lf.with_columns(
        [
            pl.col("tpep_pickup_datetime").dt.year().alias("pickup_year"),
            pl.col("tpep_pickup_datetime").dt.month().alias("pickup_month"),
            pl.col("tpep_pickup_datetime").dt.weekday().alias("pickup_weekday"),
            pl.col("tpep_pickup_datetime").dt.hour().alias("pickup_hour"),
            (
                (pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime"))
                .dt.total_minutes()
            ).alias("trip_duration_minutes"),
        ]
    )

    valid_date_filter = get_valid_date_filter(year)

    # --------------------------------------------------
    # Structural validity filters
    # These remove clearly broken or out-of-scope rows.
    # --------------------------------------------------
    structural_filter = (
        pl.col("tpep_pickup_datetime").is_not_null()
        & pl.col("tpep_dropoff_datetime").is_not_null()
        & valid_date_filter
        & (pl.col("trip_duration_minutes") > 0)
    )

    # --------------------------------------------------
    # First-pass clean modeling filters
    # These are reasonable for fare modeling, but they are
    # still modeling choices, not purely structural rules.
    # --------------------------------------------------
    clean_model_filter = (
        (pl.col("fare_amount") > MIN_FARE)
        & (pl.col("trip_distance") > MIN_TRIP_DISTANCE)
    )

    lf = lf.filter(structural_filter & clean_model_filter)

    return lf


def summarize_counts(raw_lf: pl.LazyFrame, filtered_lf: pl.LazyFrame, year: int) -> pl.DataFrame:
    raw_rows = raw_lf.select(pl.len()).collect().item()
    model_ready_rows = filtered_lf.select(pl.len()).collect().item()
    pct_retained = (model_ready_rows / raw_rows) * 100 if raw_rows > 0 else 0.0

    return pl.DataFrame(
        {
            "source_year": [year],
            "raw_rows": [raw_rows],
            "model_ready_rows": [model_ready_rows],
            "pct_retained": [pct_retained],
        }
    )


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []

    for path in INPUT_FILES:
        year = int(path.stem.split("_")[2])
        print(f"\nProcessing {path.name} ...")

        raw_lf = pl.scan_parquet(str(path))
        model_lf = build_model_ready_lf(path)

        summary = summarize_counts(raw_lf, model_lf, year)
        summaries.append(summary)

        out_path = PROCESSED_DIR / f"yellow_taxi_{year}_model_ready.parquet"

        model_df = model_lf.collect(streaming=True)
        model_df.write_parquet(out_path)

        print(f"Wrote: {out_path}")
        print(summary)

    final_summary = pl.concat(summaries, how="vertical")
    summary_path = PROCESSED_DIR / "model_ready_summary.csv"
    final_summary.write_csv(summary_path)

    print("\n=== FINAL SUMMARY ===")
    print(final_summary)
    print(f"\nWrote summary to: {summary_path}")


if __name__ == "__main__":
    main()