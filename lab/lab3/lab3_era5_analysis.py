# lab3_era5_analysis_clean.py
# ERA5 Wind Data Analysis for Berlin and Munich (Simplified & Clean)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------- Load and Prepare ---------------------- #
def load_data(file_path):
    """Load ERA5 CSV file and ensure timestamp parsing."""
    try:
        df = pd.read_csv(file_path)
        if 'timestamp' not in df.columns:
            raise ValueError("Missing 'timestamp' column.")
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        return df
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

# ---------------------- Compute Metrics ---------------------- #
def compute_wind(df):
    """Compute wind speed and direction."""
    df["wind_speed"] = np.sqrt(df["u10m"]**2 + df["v10m"]**2)
    df["wind_dir_deg"] = (np.degrees(np.arctan2(df["v10m"], df["u10m"])) + 360) % 360
    return df

def monthly_avg(df):
    """Monthly average wind speed."""
    return df.set_index("timestamp")["wind_speed"].resample("MS").mean() 

def seasonal_avg(df):
    """Seasonal average (DJF, MAM, JJA, SON)."""
    season_map = {
        12: 'DJF', 1: 'DJF', 2: 'DJF',
        3: 'MAM', 4: 'MAM', 5: 'MAM',
        6: 'JJA', 7: 'JJA', 8: 'JJA',
        9: 'SON', 10: 'SON', 11: 'SON'
    }
    df["season"] = df["timestamp"].dt.month.map(season_map)
    return df.groupby("season")["wind_speed"].mean()

# ---------------------- Analysis Prints ---------------------- #
def print_summary(name, df):
    print(f"\n=== 📄 {name} Dataset Info ===")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("\nWind Speed Summary:\n", df["wind_speed"].describe())

def print_extremes(name, df, n=5):
    daily_max = df.set_index("timestamp")["wind_speed"].resample("D").max()
    print(f"\n=== 🌬️ Top {n} Extreme Wind Days in {name} ===")
    print(daily_max.nlargest(n).to_string())

# ---------------------- Visualization ---------------------- #
def plot_visuals(monthly_berlin, monthly_munich, seasonal_berlin, seasonal_munich, berlin, munich):
    plt.style.use("seaborn-v0_8")

    # 1️⃣ Monthly average comparison
    plt.figure(figsize=(10, 5))
    plt.plot(monthly_berlin.index, monthly_berlin, label="Berlin", marker="o")
    plt.plot(monthly_munich.index, monthly_munich, label="Munich", marker="s")
    plt.title("Monthly Average Wind Speed (Berlin vs Munich)")
    plt.xlabel("Month")
    plt.ylabel("Wind Speed (m/s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2️⃣ Seasonal comparison bar chart
    plt.figure(figsize=(7, 5))
    x = np.arange(len(seasonal_berlin.index))
    plt.bar(x - 0.2, seasonal_berlin, 0.4, label="Berlin")
    plt.bar(x + 0.2, seasonal_munich, 0.4, label="Munich")
    plt.xticks(x, seasonal_berlin.index)
    plt.title("Seasonal Average Wind Speed")
    plt.ylabel("Wind Speed (m/s)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3️⃣ Wind speed histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(berlin["wind_speed"], label="Berlin", color="blue", kde=True, alpha=0.4)
    sns.histplot(munich["wind_speed"], label="Munich", color="orange", kde=True, alpha=0.4)
    plt.title("Wind Speed Distribution")
    plt.xlabel("Wind Speed (m/s)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------------- Main ---------------------- #
def main():
    print("🔹 Loading data...")
    berlin = load_data("berlin_era5_wind_20241231_20241231.csv")
    munich = load_data("munich_era5_wind_20241231_20241231.csv")
    if berlin is None or munich is None:
        return

    print("✅ Data successfully loaded.")

    print("\n🔹 Computing wind metrics...")
    berlin = compute_wind(berlin)
    munich = compute_wind(munich)
    print("✅ Wind speed & direction calculated.")

    # Monthly and seasonal means
    print("\n🔹 Calculating monthly and seasonal averages...")
    monthly_berlin = monthly_avg(berlin)
    monthly_munich = monthly_avg(munich)
    seasonal_berlin = seasonal_avg(berlin)
    seasonal_munich = seasonal_avg(munich)
    print("✅ Averages ready.")

    # Print summaries
    print_summary("Berlin", berlin)
    print_summary("Munich", munich)
    print("\n=== 📅 Berlin Monthly Averages ===")
    print(monthly_berlin)
    print("\n=== 📅 Munich Monthly Averages ===")
    print(monthly_munich)
    print("\n=== 🌦️ Seasonal Averages (Berlin) ===")
    print(seasonal_berlin)
    print("\n=== 🌦️ Seasonal Averages (Munich) ===")
    print(seasonal_munich)

    # Extreme days
    print_extremes("Berlin", berlin)
    print_extremes("Munich", munich)

    # Plot all visuals
    print("\n🔹 Generating visualizations...")
    plot_visuals(monthly_berlin, monthly_munich, seasonal_berlin, seasonal_munich, berlin, munich)
    print("✅ All plots displayed successfully!")

if __name__ == "__main__":
    main()
