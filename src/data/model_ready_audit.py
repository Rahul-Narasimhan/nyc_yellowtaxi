from pathlib import Path
import polars as pl

PROJECT_ROOT = Path(r"C:/Users/Rahul/OneDrive/Desktop/Learning/Projects/nyc_taxi")
PROCESSED_DIR = PROJECT_ROOT / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "data_quality"

FILES = [
    PROCESSED_DIR / "yellow_taxi_2023_model_ready.parquet",
    PROCESSED_DIR / "yellow_taxi_2024_model_ready.parquet",
    PROCESSED_DIR / "yellow_taxi_2025_model_ready.parquet",
]


def audit_file(path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    year = int(path.stem.split("_")[2])

    lf = (
        pl.scan_parquet(str(path))
        .with_columns(
            [
                pl.col("tpep_pickup_datetime").dt.year().alias("pickup_year"),
                pl.col("tpep_pickup_datetime").dt.month().alias("pickup_month"),
                (
                    (pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime"))
                    .dt.total_minutes()
                ).alias("trip_duration_minutes"),
            ]
        )
    )

    summary = lf.select(
        [
            pl.lit(year).alias("source_year"),
            pl.len().alias("row_count"),
            pl.col("tpep_pickup_datetime").min().alias("pickup_min"),
            pl.col("tpep_pickup_datetime").max().alias("pickup_max"),
            pl.col("tpep_dropoff_datetime").min().alias("dropoff_min"),
            pl.col("tpep_dropoff_datetime").max().alias("dropoff_max"),
            (pl.col("fare_amount") <= 0).sum().alias("fare_le_0_rows"),
            (pl.col("trip_distance") <= 0).sum().alias("trip_distance_le_0_rows"),
            (pl.col("trip_duration_minutes") <= 0).sum().alias("duration_le_0_rows"),
            (pl.col("tpep_pickup_datetime").is_null()).sum().alias("pickup_null_rows"),
            (pl.col("tpep_dropoff_datetime").is_null()).sum().alias("dropoff_null_rows"),
        ]
    ).collect()

    monthly = (
        lf.group_by(["pickup_year", "pickup_month"])
        .agg(pl.len().alias("row_count"))
        .sort(["pickup_year", "pickup_month"])
        .collect()
        .with_columns(pl.lit(year).alias("source_year"))
        .select(["source_year", "pickup_year", "pickup_month", "row_count"])
    )

    return summary, monthly


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    all_monthlies = []

    for path in FILES:
        print(f"Auditing {path.name} ...")
        summary, monthly = audit_file(path)
        all_summaries.append(summary)
        all_monthlies.append(monthly)

    df_summary = pl.concat(all_summaries, how="vertical")
    df_monthly = pl.concat(all_monthlies, how="vertical")

    summary_path = REPORT_DIR / "model_ready_audit_summary.csv"
    monthly_path = REPORT_DIR / "model_ready_audit_monthly.csv"

    df_summary.write_csv(summary_path)
    df_monthly.write_csv(monthly_path)

    print("\n=== MODEL-READY SUMMARY ===")
    print(df_summary)

    print(f"\nWrote summary to: {summary_path}")
    print(f"Wrote monthly counts to: {monthly_path}")


if __name__ == "__main__":
    main()