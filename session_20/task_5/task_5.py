"""
Session 20 - Task 5: ARIMA Model Fitting & 7-Day Temperature Forecasting
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def main():
    print("=" * 75)
    print("SESSION 20 - TASK 5: ARIMA Model Fitting & 7-Day Temperature Forecast")
    print("=" * 75)

    # 1. Load dataset
    df = pd.read_csv("session_20/dataset/ahmedabad_temperature.csv", parse_dates=["date"], index_col="date")
    series = df["temperature_celsius"]

    # 2. Fit ARIMA(p=2, d=1, q=2) based on ADF stationarity test (d=1) and ACF/PACF structure
    p, d, q = 2, 1, 2
    model = ARIMA(series, order=(p, d, q))
    model_fit = model.fit()

    print(f"\n1. ARIMA MODEL SUMMARY:")
    print(f"   • Model Order (p, d, q) : ({p}, {d}, {q})")
    print(f"   • AIC Score             : {model_fit.aic:.2f}")
    print(f"   • BIC Score             : {model_fit.bic:.2f}")

    # 3. Forecast Next 7 Days
    forecast_steps = 7
    forecast_res = model_fit.get_forecast(steps=forecast_steps)
    forecast_values = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int(alpha=0.05)

    # Create forecast date index
    last_date = series.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq="D")
    forecast_df = pd.DataFrame({
        "forecast_temp": forecast_values.values,
        "lower_ci": conf_int.iloc[:, 0].values,
        "upper_ci": conf_int.iloc[:, 1].values
    }, index=forecast_dates)

    print(f"\n2. PREDICTED 7-DAY TEMPERATURE FORECAST (°C):")
    print("-" * 65)
    for date, row in forecast_df.iterrows():
        print(f"   • {date.strftime('%Y-%m-%d (%a)')} : {row['forecast_temp']:.2f}°C  [95% CI: {row['lower_ci']:.2f}°C - {row['upper_ci']:.2f}°C]")
    print("-" * 65)

    # 4. Plot Historical Series (Last 60 Days) + 7-Day Forecast
    historical_subset = series.iloc[-60:]

    plt.figure(figsize=(10, 6))
    plt.plot(historical_subset.index, historical_subset.values, color="#1f77b4", lw=2, label="Observed Historical Temperature")
    plt.plot(forecast_df.index, forecast_df["forecast_temp"], color="#d62728", lw=2.5, marker="o", label="7-Day ARIMA(2,1,2) Forecast")
    plt.fill_between(forecast_df.index, forecast_df["lower_ci"], forecast_df["upper_ci"], color="#d62728", alpha=0.2, label="95% Confidence Interval")

    plt.title("Ahmedabad Temperature - ARIMA(2,1,2) 7-Day Forecast", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Temperature (°C)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()

    output_img = "session_20/task_5/arima_forecast_plot.png"
    plt.savefig(output_img, dpi=300)
    plt.close()

    print(f"\n3. Plot saved successfully to: {output_img}")
    print("=" * 75)

if __name__ == "__main__":
    main()
