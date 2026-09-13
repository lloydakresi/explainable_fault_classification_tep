from inference import df_causal_analysis
import numpy as np
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import tigramite.plotting as tp
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

def plot_current_time_effects(df_causal_analysis, tau_max=80, pc_alpha=0.05, min_strength=0.1, figsize=(12, 8)):
    """
    Runs PCMCI and visualizes a 'clean' version of drivers affecting time (t).
    """
    var_names = list(df_causal_analysis.columns)
    tigramite_df = pp.DataFrame(df_causal_analysis.values, var_names=var_names)

    # 1. Run PCMCI (using run_pcmci for cleaner lagged discovery)
    cond_ind_test = ParCorr()
    pcmci = PCMCI(tigramite_df, cond_ind_test=cond_ind_test)
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)

    graph = results['graph']
    val_matrix = results['val_matrix']
    num_vars = len(var_names)

    # 2. Extract significant drivers only
    G = nx.DiGraph()
    edge_data = []

    for j in range(num_vars):         # Target (t)
        for i in range(num_vars):     # Driver (t-tau)
            if i == j: continue       # Ignore self-loops for a cleaner graph

            # Find the strongest lag for this pair
            strengths = np.abs(val_matrix[i, j, 1:])
            max_tau = np.argmax(strengths) + 1
            strength = val_matrix[i, j, max_tau]

            if graph[i, j, max_tau] == '-->' and abs(strength) >= min_strength:
                G.add_edge(var_names[i], var_names[j], weight=abs(strength), label=f"lag {max_tau}")
                edge_data.append({
                    'Driver': var_names[i],
                    'Target': var_names[j],
                    'Lag': max_tau,
                    'Strength': round(strength, 3)
                })

    # 3. Clean Visualization (The Anti-Hairball Logic)
    plt.figure(figsize=figsize)

    # spring_layout with k increases the distance between nodes
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

    # Scale edges: thicker = stronger causal link
    weights = [G[u][v]['weight'] * 8 for u, v in G.edges()]

    # Draw nodes with high contrast
    nx.draw_networkx_nodes(G, pos, node_size=2500, node_color='#3498db', edgecolors='white', linewidths=2)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='white', font_weight='bold')

    # Draw curved edges to avoid overlapping lines
    nx.draw_networkx_edges(G, pos, width=weights, arrowsize=25,
                           edge_color='#95a5a6', alpha=0.7,
                           connectionstyle='arc3,rad=0.15')

    plt.title(f"Primary Causal Drivers of System at Time (t)\n(Threshold: {min_strength})", pad=20)
    plt.axis('off')
    plt.show()

    # 4. Human-Interpretable Output (The "Narrative" Table)
    df_summary = pd.DataFrame(edge_data).sort_values(by='Lag', key=abs, ascending=False)

    print("\n--- HUMAN INTERPRETABLE SUMMARY ---")
    if not df_summary.empty:
        for _, row in df_summary.iterrows():
            direction = "increases" if row['Strength'] > 0 else "decreases"
            print(f"• {row['Driver']} (from {row['Lag']} steps ago) {direction} {row['Target']} (Strength: {row['Strength']})")
    else:
        print("No significant causal relationships found above the threshold.")


    return results, df_summary


'''
results, df_summary = plot_current_time_effects(
    df_causal_analysis=df_causal_analysis,
    tau_max=20,
    pc_alpha=0.1,
    min_strength=0.2,
)
'''
