#create json cer for prompting
from inference import logits,  most_important_encoder_vars, most_important_temps, df_cer, X_dict
import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
import pandas as pd
import json

probabilities = F.softmax(logits, dim=-1)
confidence, prediction = torch.max(probabilities, dim=-1)
def get_time_series_metrics(df):
    results = {}
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        y = df[col].dropna().values
        if len(y) < 3: continue  # Need at least 3 points for acceleration/R2

        x = np.arange(len(y))

        # 1. Linear Regression (Slope and R2)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r2 = r_value**2


        # 2. Normalized Slope (Slope relative to the average value)
        mean_val = np.mean(y)
        norm_slope = slope / mean_val if mean_val != 0 else 0

        # 3. Relative Change (Total % change from start to end)
        rel_change = (y[-1] - y[0]) / y[0] if y[0] != 0 else 0

        # 4. Volatility (Standard deviation of percentage changes)
        returns = pd.Series(y).pct_change().dropna()
        volatility = returns.std()

        # 5. Acceleration (Mean of the second derivative/change in slope)
        velocity = np.diff(y)
        acceleration = np.mean(np.diff(velocity))

        results[col] = {
            "slope": round(float(slope), 4),
            "normalized_slope": round(float(norm_slope), 4),
            "relative_change": round(float(rel_change), 4),
            "volatility": round(float(volatility), 4),
            "acceleration": round(float(acceleration), 4),
            "r2": round(float(r2), 4)
        }
    return results




CER = {
    "X_dict":X_dict,
    "fault":{
        #json can't serialize numpy floats
        "id":float(prediction.item()),
        "confidence": float(confidence.item() * 100),
    },
    "top_variables": most_important_encoder_vars,
    "top_lags": most_important_temps,
    "trend_metrics": get_time_series_metrics(df_cer),
}

CER_JSON = json.dumps(CER, indent=4)
print(CER)
