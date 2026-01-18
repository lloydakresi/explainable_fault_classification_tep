import pandas as pd
import torch
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

demo_path = "demo_sequences/fault_15.parquet"
df = pd.read_parquet(demo_path)
future_unknown_inputs = [f"xmeas_{i}" for i in range(1, 42)]
future_known_inputs = [f"xmv_{i}" for i in range(1, 12)]
features = future_unknown_inputs + future_known_inputs
df[features] = df[features].astype(np.float32)
df["group_id"] = df["simulationRun"].astype(str) + "_" + df["faultNumber"].astype(str)

params = torch.load("dataset/dataset_parameters.pt", weights_only=False)
sample_set = TimeSeriesDataSet.from_parameters(params, df, predict=True)
sample_dataloader = sample_set.to_dataloader(train=False, batch_size=1)
sample_x, sample_y = next(iter(sample_dataloader))


path = "cpu_compat_checkpoint.ckpt"

tft = TemporalFusionTransformer.load_from_checkpoint(
    path,
    map_location=torch.device('cpu'),
    strict=False
)

tft.eval()
with torch.no_grad():
    '''
    raw_predictions= tft.predict(sample_dataloader, mode="raw", return_x=True)
    interpretation = tft.interpret_output(raw_predictions.output, reduction="sum")
    contributions = interpretation["attention"]
    print(len(contributions/contributions.sum()))
    #(contributions/contributions.sum()) * 100
    tft.plot_interpretation(interpretation)
    plt.show()
    '''
    print(sample_x)
    print("-------------------------------------------")
    prediction_input = tft(sample_x)
    logits = prediction_input.prediction
    predictions = tft.interpret_output(prediction_input, reduction="none")
    temporal_attention = predictions["attention"]
    encoder_attention = predictions["encoder_variables"]
    decoder_attention = predictions["decoder_variables"]
