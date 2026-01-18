import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import CrossEntropy
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl

def train():
    df_training = pd.read_feather("dataset/df_training.feather")
    df_testing = pd.read_feather("dataset/df_testing.feather")


    params = torch.load("dataset_parameters.pt")
    training_dataset = TimeSeriesDataSet.from_parameters(params, df_training)
    training_dataloader = training_dataset.to_dataloader(batch_size=32, num_workers=4)

    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        df_testing,
        stop_randomization=True,
    )
    validation_dataloader = validation_dataset.to_dataloader(train=False, batch_size=32*10, num_workers=2)


    pl.seed_everything(42)

    early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
    )

    lr_logger = LearningRateMonitor()

    logger = TensorBoardLogger("tepBenchmarkLog")
    trainer = pl.Trainer(
        max_epochs=25,
        accelerator="auto",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stopping],
        logger=logger,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.03,
        hidden_size=9,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=CrossEntropy(),
        optimizer="ranger",
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")


    trainer.fit(
        tft,
        train_dataloaders=training_dataloader,
        val_dataloaders=validation_dataloader,
    )


if __name__ == "__main__":
    # This block only runs in the main process
    train()
