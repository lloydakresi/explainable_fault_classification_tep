from explainable_fault_classification_tep.causality.pcmci import plot_current_time_effects
from inference import df_causal_analysis
import math

#find most likely parent variable
#hypothesis is that they're the driver variable with the highest mode
_, df_summary = plot_current_time_effects(
    df_causal_analysis=df_causal_analysis
)
#define weights for each row
def calculate_weight(strength, lag, lam=0.02):
    # We use abs(strength) for magnitude
    # we subtract 1 from lag so that Lag 1 is the 'gold standard'
    magnitude = abs(strength)
    penalty = math.exp(lam * (lag))
    return magnitude * penalty

df_summary["Weight"] = df_summary.apply(
    lambda x: calculate_weight(x["Strength"], x["Lag"]), axis=1
)
parent_variables = df_summary["Driver"].mode().values

#if there is more than one candidate, select the parent with the highest strength
subset = df_summary[df_summary["Driver"].isin(parent_variables)]
best_index = subset["Weight"].idxmax()
parent_variable = subset.loc[[best_index]].to_dict(orient="records")[0]
print(parent_variable)
graph_data = df_summary.to_dict(orient="records")
visited = []


def build_causal_path(current_driver, records, visited=None, parent_lag=None):
    """
    Recursively builds and prints a causal path using nodes structured as:
    {'Driver', 'Target', 'Lag', 'Strength', 'Weight'}

    Only children with lag <= parent_lag are considered.
    If parent_lag is None, this is the starting node and all children are allowed.
    """
    if visited is None:
        visited = []

    if current_driver in visited:
        print("Endmmmmm")
        return
    visited.append(current_driver)

    # Filter children: not visited and lag <= parent_lag (if parent_lag is not None)
    children = [
        node for node in records
        if node['Driver'] == current_driver
        and node['Target'] not in visited
        and (parent_lag is None or node['Lag'] <= parent_lag)
    ]

    if not children:
        print("End")
        return

    # Pick child with highest Weight among allowed children
    best_child = max(children, key=lambda x: x['Weight'])

    print(f"{best_child['Driver']} -> {best_child['Target']} "
          f"(Weight: {best_child['Weight']:.3f}, Lag: {best_child['Lag']})")

    # Recurse: now parent_lag becomes the lag of this child
    build_causal_path(best_child['Target'], records, visited, parent_lag=best_child['Lag'])


build_causal_path(parent_variable["Driver"], graph_data, visited)



'''
#alternative algorithm
from inference import sorted_temp_attention_dict, sorted_encoder_attention_dict, sorted_decoder_attention_dict

'''
