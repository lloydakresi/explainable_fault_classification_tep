import pandas as pd
import torch
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from itertools import islice
X_dict = {
'XMEAS_1':'A_feed_stream',
'XMEAS_2':'D_feed_stream',
'XMEAS_3':'E_feed_stream',
'XMEAS_4':'Total_fresh_feed_stripper',
'XMEAS_5':'Recycle_flow_into_rxtr',
'XMEAS_6':'Reactor_feed_rate',
'XMEAS_7':'Reactor_pressure',
'XMEAS_8':'Reactor_level',
'XMEAS_9':'Reactor_temp',
'XMEAS_10':'Purge_rate',
'XMEAS_11':'Separator_temp',
'XMEAS_12':'Separator_level',
'XMEAS_13':'Separator_pressure',
'XMEAS_14':'Separator_underflow',
'XMEAS_15':'Stripper_level',
'XMEAS_16':'Stripper_pressure',
'XMEAS_17':'Stripper_underflow',
'XMEAS_18':'Stripper_temperature',
'XMEAS_19':'Stripper_steam_flow',
'XMEAS_20':'Compressor_work',
'XMEAS_21':'Reactor_cooling_water_outlet_temp',
'XMEAS_22':'Condenser_cooling_water_outlet_temp',
'XMEAS_23':'Composition_of_A_rxtr_feed',
'XMEAS_24':'Composition_of_B_rxtr_feed',
'XMEAS_25':'Composition_of_C_rxtr_feed',
'XMEAS_26':'Composition_of_D_rxtr_feed',
'XMEAS_27':'Composition_of_E_rxtr_feed',
'XMEAS_28':'Composition_of_F_rxtr_feed',
'XMEAS_29':'Composition_of_A_purge',
'XMEAS_30':'Composition_of_B_purge',
'XMEAS_31':'Composition_of_C_purge',
'XMEAS_32':'Composition_of_D_purge',
'XMEAS_33':'Composition_of_E_purge',
'XMEAS_34':'Composition_of_F_purge',
'XMEAS_35':'Composition_of_G_purge',
'XMEAS_36':'Composition_of_H_purge',
'XMEAS_37':'Composition_of_D_product',
'XMEAS_38':'Composition_of_E_product',
'XMEAS_39':'Composition_of_F_product',
'XMEAS_40':'Composition_of_G_product',
'XMEAS_41':'Composition_of_H_product',
'XMV_1':'D_feed_flow_valve',
'XMV_2':'E_feed_flow_valve',
'XMV_3':'A_feed_flow_valve',
'XMV_4':'Total_feed_flow_stripper_valve',
'XMV_5':'Compressor_recycle_valve',
'XMV_6':'Purge_valve',
'XMV_7':'Separator_pot_liquid_flow_valve',
'XMV_8':'Stripper_liquid_product_flow_valve',
'XMV_9':'Stripper_steam_valve',
'XMV_10':'Reactor_cooling_water_flow_valve',
'XMV_11':'Condenser_cooling_water_flow_valve',
   }
X_dict = {
    key.lower():value for (key, value) in X_dict.items()
}

def prepare_sample_sequence(path, X_dict=X_dict):
    df = pd.read_parquet(path)
    future_unknown_inputs = [f"xmeas_{i}" for i in range(1, 42)]
    future_known_inputs = [f"xmv_{i}" for i in range(1, 12)]
    interpretable_future_unknown_inputs = [v for (k, v) in X_dict.items() if k.startswith("xmeas_") ]
    interpretable_future_known_inputs = [v for (k, v) in X_dict.items() if k.startswith("xmv_") ]
    features = future_unknown_inputs + future_known_inputs
    df[features] = df[features].astype(np.float32)
    df["group_id"] = df["simulationRun"].astype(str) + "_" + df["faultNumber"].astype(str)
    return df, future_known_inputs, future_unknown_inputs


df, future_known_variables, future_unknown_variables = prepare_sample_sequence("demo_sequences/fault_09.parquet")
decoder_variables = future_known_variables + ["relative_time_idx"]
encoder_variables = decoder_variables + future_unknown_variables
#load the TimeSeriesDataSet parameters
params = torch.load("dataset/dataset_parameters.pt", weights_only=False)
#create a TimeSeriesDataset using the parameters from training data and the testing data
sample_set = TimeSeriesDataSet.from_parameters(params, df, predict=True)
#convert into dataloader
sample_dataloader = sample_set.to_dataloader(train=False, batch_size=1)
#convert into iterable
it = iter(sample_dataloader)
#first sequence
sample_x, sample_y = next(it)
#second sequence
sample_x2, sample_y2 = next(it)
sample_x22, sample_y22 = next(it)

#path to cpu-compatible model parameters
path = "cpu_compat_checkpoint.ckpt"

#load model using the path
tft = TemporalFusionTransformer.load_from_checkpoint(
    path,
    map_location=torch.device('cpu'),
    strict=False
)

#set model to evaluation mode
tft.eval()
with torch.no_grad():
    #refactor next 19 lines into function
    prediction_input = tft(sample_x2)
    #obtain the confidence from the code below
    logits = prediction_input.prediction
    predictions = tft.interpret_output(prediction_input, reduction="none")
    print(logits)

    temporal_attention_timestamps = list(range(1, 251))

    def feature_contributions(prediction_attention):
        keys = prediction_attention.keys()
        interpretable_attention = {}
        for key in keys:
            prediction = prediction_attention[key]
            prediction = (prediction/prediction.sum())*100
            prediction = prediction.tolist()[0]
            interpretable_attention[key] = prediction
        return interpretable_attention

    attention = feature_contributions(predictions)
    temporal_attention = attention["attention"]
    encoder_attention = attention["encoder_variables"]
    decoder_attention = attention["decoder_variables"]


    #refactor lines below: too repetitive
    temp_attention_dict = dict(zip(reversed(temporal_attention_timestamps), temporal_attention))
    sorted_temp_attention_dict = {k:v for k,v in sorted(temp_attention_dict.items(), key=lambda item:item[1], reverse=True)}
    most_important_temps = dict(islice(sorted_temp_attention_dict.items(), 10))

    encoder_attention_dict = dict(zip(encoder_variables, encoder_attention))
    sorted_encoder_attention_dict = {k:v for k,v in sorted(encoder_attention_dict.items(), key=lambda item:item[1], reverse=True)}
    most_important_encoder_vars = dict(islice(sorted_encoder_attention_dict.items(), 10))
    int_most_important_enc_vars = {
        X_dict[k] : v for (k, v) in most_important_encoder_vars.items()
    }

    decoder_attention_dict = dict(zip(decoder_variables, decoder_attention))
    sorted_decoder_attention_dict = {k:v for k,v in sorted(decoder_attention_dict.items(), key=lambda item:item[1], reverse=True)}
    most_important_decoder_vars = dict(islice(sorted_decoder_attention_dict.items(), 5))
    int_most_important_dec_vars = {
        X_dict[k]:v for (k, v) in most_important_decoder_vars.items()
    }

    causal_analysis_variables = most_important_encoder_vars.keys()

    encoder_df = pd.DataFrame(
        sample_x2["encoder_cont"].detach().cpu().numpy()[0],
        columns=sample_set.reals
    )
    df_cer = encoder_df[causal_analysis_variables]
