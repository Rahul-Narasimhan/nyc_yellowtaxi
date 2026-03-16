from pathlib import Path
import polars as pl

PROJECT_ROOT = Path(r"C:/Users/Rahul/OneDrive/Desktop/Learning/Projects/nyc_taxi")
INTERIM_DIR = PROJECT_ROOT / "interim"
REPORT_DIR = PROJECT_ROOT / "reports" / "data_quality"

FILES = [
    INTERIM_DIR / "yellow_taxi_2023_normalized.parquet",
    INTERIM_DIR / "yellow_taxi_2024_normalized.parquet",
    INTERIM_DIR / "yellow_taxi_2025_normalized.parquet",
]

KEY_COLUMNS = [
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
    "total_amount",
    "airport_fee",
    "cbd_congestion_fee",
]

def audit_file(path: Path) -> pl.DataFrame:
    year = int(path.stem.split("_")[2])

    lf = (
        pl.scan_parquet(str(path))
        .with_columns([
            pl.col("tpep_pickup_datetime").dt.year().alias("pickup_year"),
            pl.col("tpep_pickup_datetime").dt.month().alias("pickup_month"),
            (
                (pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime"))
                .dt.total_minutes()
            ).alias("trip_duration_minutes")
        ])
    )

    summary = lf.select([
        pl.lit(year).alias("source_year"),
        pl.len().alias("row_count"),

        pl.col("tpep_pickup_datetime").min().alias("pickup_min"),
        pl.col("tpep_pickup_datetime").max().alias("pickup_max"),
        pl.col("tpep_dropoff_datetime").min().alias("dropoff_min"),
        pl.col("tpep_dropoff_datetime").max().alias("dropoff_max"),

        pl.col("fare_amount").null_count().alias("fare_amount_nulls"),
        pl.col("trip_distance").null_count().alias("trip_distance_nulls"),
        pl.col("passenger_count").null_count().alias("passenger_count_nulls"),
        pl.col("payment_type").null_count().alias("payment_type_nulls"),

        (pl.col("fare_amount") < 0).sum().alias("negative_fare_rows"),
        (pl.col("fare_amount") == 0).sum().alias("zero_fare_rows"),

        (pl.col("trip_distance") < 0).sum().alias("negative_trip_distance_rows"),
        (pl.col("trip_distance") == 0).sum().alias("zero_trip_distance_rows"),

        (pl.col("trip_duration_minutes") < 0).sum().alias("negative_duration_rows"),
        (pl.col("trip_duration_minutes") == 0).sum().alias("zero_duration_rows"),

        (pl.col("passenger_count") < 0).sum().alias("negative_passenger_count_rows"),
        (pl.col("passenger_count") == 0).sum().alias("zero_passenger_count_rows"),
        (pl.col("passenger_count") > 6).sum().alias("passenger_count_gt_6_rows"),

        (pl.col("fare_amount") > 300).sum().alias("fare_gt_300_rows"),
        (pl.col("trip_distance") > 100).sum().alias("trip_distance_gt_100_rows"),
        (pl.col("trip_duration_minutes") > 240).sum().alias("duration_gt_240_rows"),
    ]).collect()

    monthly = (
        lf.group_by(["pickup_year", "pickup_month"])
        .agg([
            pl.len().alias("row_count"),
            (pl.col("fare_amount") < 0).sum().alias("negative_fare_rows"),
            (pl.col("trip_distance") <= 0).sum().alias("nonpositive_trip_distance_rows"),
            (pl.col("trip_duration_minutes") <= 0).sum().alias("nonpositive_duration_rows"),
        ])
        .sort(["pickup_year", "pickup_month"])
        .collect()
        .with_columns(pl.lit(year).alias("source_year"))
        .select([
            "source_year", "pickup_year", "pickup_month", "row_count",
            "negative_fare_rows", "nonpositive_trip_distance_rows", "nonpositive_duration_rows"
        ])
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

    summary_path = REPORT_DIR / "normalized_data_quality_summary.csv"
    monthly_path = REPORT_DIR / "normalized_data_quality_monthly.csv"

    df_summary.write_csv(summary_path)
    df_monthly.write_csv(monthly_path)

    print("\n=== YEARLY SUMMARY ===")
    print(df_summary)

    print(f"\nWrote yearly summary to: {summary_path}")
    print(f"Wrote monthly summary to: {monthly_path}")

if __name__ == "__main__":
    main()