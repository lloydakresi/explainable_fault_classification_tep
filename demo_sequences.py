import pandas as pd
from pathlib import Path

demo_dataset = pd.read_feather("df_demo.feather")
demo_group = demo_dataset.groupby(["faultNumber"])
df_normal = demo_group.get_group((0,))




out_dir = Path("demo_sequences")
out_dir.mkdir(exist_ok=True)

for i in range(0, 21):
    df_onset = df_normal[df_normal["simulationRun"] == 51]
    df_aftermath = df_normal[df_normal["simulationRun"] == 53]

    df_fault = demo_group.get_group((i,))
    fault_steady = df_fault[df_fault["simulationRun"] == 52]

    sequence = pd.concat(
        [
            df_onset.assign(phase="onset"),
            fault_steady.assign(phase="steady"),
            df_aftermath.assign(phase="aftermath"),
        ],
        ignore_index=True,
    )

    sequence = sequence.reset_index(drop=True)
    sequence["sample"] = sequence.index + 1

    # --- save ---
    fault_id = pd.unique(df_fault["faultNumber"])[0]
    if fault_id == 0:
        out_path = out_dir / f"normal.parquet"
    else:
        out_path = out_dir / f"fault_{fault_id:02d}.parquet"

    sequence.to_parquet(out_path, index=False)

    print(f"Saved {out_path}")
