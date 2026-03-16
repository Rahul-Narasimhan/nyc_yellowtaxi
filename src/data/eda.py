import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
from pathlib import Path

# 1. SETUP PATHS (Adjust if your PROJECT_ROOT differs)
PROJECT_ROOT = Path(r"C:/Users/Rahul/OneDrive/Desktop/Learning/Projects/nyc_taxi")
PROCESSED_DIR = PROJECT_ROOT / "processed"
OUTPUT_DIR = PROJECT_ROOT / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# 2. LOAD A REPRESENTATIVE SAMPLE (Using your existing sampling logic)
files = [PROCESSED_DIR / f"yellow_taxi_{y}_model_ready.parquet" for y in [2023, 2024, 2025]]
lf = pl.scan_parquet([str(f) for f in files])

# Sample 100k rows for fast visualization
df_eda = lf.select([
    "fare_amount", "trip_distance", "passenger_count", 
    "pickup_hour", "pickup_weekday", "PULocationID"
]).collect().sample(n=100_000, seed=42).to_pandas()

# 3. PROFESSIONAL PLOTTING CONFIG
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('NYC Yellow Taxi: Feature Distributions & Data Quality Analysis', fontsize=20, fontweight='bold')

# --- PLOT 1: Target Variable (Fare Amount) ---
# Note: Using log-scale because fares are heavily right-skewed
sns.histplot(df_eda['fare_amount'], bins=100, ax=axes[0, 0], kde=True, color='teal')
axes[0, 0].set_title('Fare Amount Distribution (Log Scale)', fontsize=14)
axes[0, 0].set_yscale('log')
axes[0, 0].set_xlabel('Fare ($)')

# --- PLOT 2: Trip Distance ---
# Capping at 20 miles for visibility (handling outliers)
sns.histplot(df_eda[df_eda['trip_distance'] < 20]['trip_distance'], bins=50, ax=axes[0, 1], kde=True, color='darkorange')
axes[0, 1].set_title('Trip Distance (Capped at 20mi)', fontsize=14)
axes[0, 1].set_xlabel('Distance (miles)')

# --- PLOT 3: Passenger Count (Discrete) ---
sns.countplot(data=df_eda, x='passenger_count', ax=axes[0, 2], palette='viridis')
axes[0, 2].set_title('Passenger Count Frequency', fontsize=14)

# --- PLOT 4: Hourly Demand (Cyclical Pattern) ---
sns.kdeplot(df_eda['pickup_hour'], fill=True, ax=axes[1, 0], color='purple')
axes[1, 0].set_title('Demand by Hour of Day', fontsize=14)
axes[1, 0].set_xticks(range(0, 24, 2))

# --- PLOT 5: Weekly Demand ---
day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
sns.countplot(data=df_eda, x='pickup_weekday', ax=axes[1, 1], palette='magma')
axes[1, 1].set_title('Demand by Day of Week', fontsize=14)
axes[1, 1].set_xticklabels(day_labels)

# --- PLOT 6: Spatial Concentration (Top Pickup IDs) ---
top_locations = df_eda['PULocationID'].value_counts().head(10)
sns.barplot(x=top_locations.index, y=top_locations.values, ax=axes[1, 2], palette='coolwarm', order=top_locations.index)
axes[1, 2].set_title('Top 10 Pickup Location IDs', fontsize=14)
axes[1, 2].set_xlabel('LocationID')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plot_path = OUTPUT_DIR / "feature_distributions_dashboard.png"
plt.savefig(plot_path, dpi=300)
print(f"EDA Dashboard saved to: {plot_path}")
plt.show()