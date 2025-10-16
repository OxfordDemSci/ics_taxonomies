def make_weight_plot():
    """
    Produce frequency and reassignment-likelihood plots over BERT probabilities.
    Fixes boundary inclusion and label inversion.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'Helvetica'
    df = pd.read_csv('../data/final/enhanced_ref_data.csv')
    df['topic1_reassign'] = df['topic1_reassign'].fillna(0)
    ba_rgb2 = ['#41558c', '#E89818', '#CF202A']

    # --- Robust binning over [0,1] inclusive ---
    bins = np.linspace(0, 1 + 1e-8, 21)  # ensure 1.0 is included
    labels = [f'{bins[i]:.2f}-{bins[i+1]:.2f}' for i in range(len(bins) - 1)]

    df['bin'] = pd.cut(
        df['bert_prob'].clip(0, 1),
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False
    )

    grouped = df.groupby('bin', observed=False)
    holder = grouped['bert_prob'].size().to_frame('N')
    # compute true reassignment likelihood
    holder['Prob'] = grouped['topic1_reassign'].apply(lambda s: (s == 1).mean())
    holder = holder.fillna(0)

    base_rate = (df['topic1_reassign'] == 1).mean()
    print(base_rate)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)

    holder['N'].plot(kind='bar', color=ba_rgb2[0], edgecolor='k', ax=ax1, width=1)
    holder['Prob'].plot(kind='bar', color=ba_rgb2[1], edgecolor='k', ax=ax2, width=1)

    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax1.grid(False)
    ax2.grid(False)
    ax2.spines[['left', 'top']].set_visible(False)
    sns.despine(ax=ax1)

    ax1.set_title('a.', loc='left', fontsize=18, y=1.025)
    ax2.set_title('b.', loc='left', fontsize=18, y=1.025)
    ax1.set_xlabel('Original BERT Weight', fontsize=15)
    ax2.set_xlabel('Original BERT Weight', fontsize=15)
    ax1.set_ylabel('Frequency', fontsize=15)
    ax2.set_ylabel('Likelihood of Reassignment', fontsize=15)

    mpl.rcParams['font.family'] = 'Graphik'
    mpl.rcParams.update({'text.usetex': False, "svg.fonttype": 'none'})

    figure_path = os.path.join(os.getcwd(), '..', 'figures')
    os.makedirs(figure_path, exist_ok=True)
    for ext in ['svg', 'pdf', 'png']:
        plt.savefig(os.path.join(figure_path, f'bert_weights.{ext}'),
                    bbox_inches='tight', dpi=600 if ext == 'png' else None)


def plot_heatmap_mds_K7(
    reassignment_path="../data/reassignments/Reassignment.xlsx",
    raw_path="../data/raw/raw_ref_ics_data.xlsx",
    figsize=(12, 6),
    max_features=2000,
):
    """
    Load data, compute hierarchical clusters of topic centroids, and plot
    a 1×2 figure with:
        (a) cosine-distance heatmap (ordered by cluster, with outer spines),
        (b) MDS projection with semantic cluster labels and colours from 'Spectral'.
    """
    import os
    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import silhouette_score, silhouette_samples
    from sklearn.manifold import MDS
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'Helvetica'

    # --- 1. Load and merge data ------------------------------------------------
    df_re = pd.read_excel(reassignment_path, sheet_name="ICSs")
    df_raw = pd.read_excel(raw_path)

    cols = [
        "1. Summary of the impact",
        "2. Underpinning research",
        "3. References to the research",
        "4. Details of the impact",
        "5. Sources to corroborate the impact",
    ]
    df_raw["merged"] = df_raw[cols].fillna("").agg(" ".join, axis=1)

    df_m = pd.merge(
        df_re,
        df_raw[["REF impact case study identifier", "merged"]],
        on="REF impact case study identifier",
        how="left",
    )

    # --- 2. TF–IDF representation ----------------------------------------------
    tfidf = TfidfVectorizer(max_features=max_features, stop_words="english")
    X = tfidf.fit_transform(df_m["merged"])
    df_m["vec"] = list(X.toarray())

    # --- 3. Compute topic centroids --------------------------------------------
    centroids = (
        df_m.groupby("Assigned Topic")["vec"]
        .apply(lambda vs: np.mean(np.vstack(vs), axis=0))
        .to_dict()
    )
    topic_ids = sorted(centroids.keys())
    C = np.vstack([centroids[t] for t in topic_ids])

    # --- 4. Distance matrix and hierarchical clustering ------------------------
    D_cond = pdist(C, metric="cosine")
    D_sq = squareform(D_cond)
    Z = linkage(D_cond, method="ward")

    # --- 5. Cluster assignment for K = 7 --------------------------------------
    K = 7
    labels = fcluster(Z, t=K, criterion="maxclust")

    # --- 6. Silhouette statistics ---------------------------------------------
    sil_avg = silhouette_score(C, labels, metric="cosine")
    sil_vals = silhouette_samples(C, labels, metric="cosine")

    # --- 7. Reorder distances by cluster label --------------------------------
    order = [t for _, t in sorted(zip(labels, topic_ids))]
    idx_order = [topic_ids.index(t) for t in order]
    D_ord = D_sq[np.ix_(idx_order, idx_order)]

    # --- 8. MDS embedding ------------------------------------------------------
    coords = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=0,
        n_init=4,  # silence FutureWarning
    ).fit_transform(D_sq)

    # --- 9. Semantic labels & Spectral colours --------------------------------
    semantic_labels = {
        1: "Cancer",
        2: "Health",
        3: "Technology",
        4: "Environment",
        5: "Culture",
        6: "Education",
        7: "Society",
    }

    spectral = cm.get_cmap("Spectral", 7)
    cluster_colors = [spectral(i / 6) for i in range(7)]  # evenly spaced

    # --- 10. Plot --------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # (a) Heatmap (no cell edges, only outer spines)
    sns.heatmap(D_ord, cmap="Spectral", cbar=False, ax=ax1, square=True)
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("black")

    ax1.set_title("a.", loc="left", fontsize=16, fontweight="bold")
    ax1.set_xlabel("Topics (ordered by cluster)", fontsize=12)
    ax1.set_ylabel("Topics (ordered by cluster)", fontsize=12)

    # (b) MDS Projection
    unique_labels = np.unique(labels)
    for i, c in enumerate(unique_labels):
        idx = labels == c
        ax2.scatter(
            coords[idx, 0],
            coords[idx, 1],
            label=semantic_labels.get(c, f"C{c}"),
            s=50,
            color=cluster_colors[i],
            edgecolor="black",
            linewidth=0.5,
        )

    ax2.set_title("b.", loc="left", fontsize=16, fontweight="bold")
    ax2.set_xlabel("MDS-1", fontsize=14)
    ax2.set_ylabel("MDS-2", fontsize=14)

    # Clean, subtle gridlines
    ax2.grid(True, which="major", color="lightgray", linestyle="--", linewidth=0.6, alpha=0.7)

    # Semantic legend
    ax2.legend(
        loc="lower left",
        ncols=2,
        fontsize=9,
        frameon=True,
        edgecolor="black",
        title_fontsize=10,
    )

    figure_path = os.path.join(os.getcwd(), '..', 'figures')
    os.makedirs(figure_path, exist_ok=True)
    for ext in ['svg', 'pdf', 'png']:
        plt.savefig(os.path.join(figure_path, f'k7_clustering.{ext}'),
                    bbox_inches='tight', dpi=600 if ext == 'png' else None)

    # --- 11. Return results ----------------------------------------------------
    return {
        "labels": labels,
        "silhouette_avg": sil_avg,
        "silhouette_values": sil_vals,
        "topic_ids": topic_ids,
    }


def visualize_topic_models():
    import os
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.offsetbox import AnchoredText
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'Helvetica'
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5),
                            sharex=True)

    meta = pd.read_csv(os.path.join(os.getcwd(),
                                    '..',
                                    'data',
                                    'topic_modelled',
                                    'metadata.csv'))

    for ax, col in zip(axs, ['columns124', 'column12345']):
        # Scatter plot
        scatter = ax.scatter(
            x=meta[meta['columns'] == col]['topics_count'],
            y=meta[meta['columns'] == col]['outliers_count'],
            s=meta[meta['columns'] == col]['silhouette_score'] * 2500 - 1000,
            c=meta[meta['columns'] == col]['silhouette_score'] * 2500 - 1000,
            edgecolor='k'
        )
        ax.set_ylim(1800, 3200)
        ax.set_xlim(60, 200)

    # Add colorbar
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cmap = cm.viridis  # You can change the colormap as needed
    norm = Normalize(vmin=meta['silhouette_score'].min(),
                     vmax=meta['silhouette_score'].max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Add colorbar to the last subplot
    plt.colorbar(sm, cax=cax, label='Silhouette Score')

    # Set titles and labels
    axs[0].set_title('a.', loc='left', fontsize=16)
    axs[1].set_title('b.', loc='left', fontsize=16)
    axs[0].set_ylabel('Number of Outliers')
    axs[0].set_xlabel('Number of Topics')
    axs[1].set_xlabel('Number of Topics')
    at = AnchoredText(
        "cols = ['1. Summary of the impact',\n" +
        "            '2. Underpinning research',\n" +
        "            '3. References to the research',\n" +
        "            '4. Details of the impact',\n" +
        "            '5. Sources to corroborate the impact']",
        prop=dict(size=8), frameon=True, loc='upper right')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.4")
    axs[1].add_artist(at)

    at = AnchoredText(
        "cols = ['1. Summary of the impact',\n" +
        "            '2. Underpinning research',\n" +
        "            '4. Details of the impact']",
        prop=dict(size=8), frameon=True, loc='upper right')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.4")
    axs[0].add_artist(at)

    # First subplot
    max_silhouette_idx = meta[meta['columns'] == 'columns124']['silhouette_score'].idxmax()
    max_silhouette_values = meta.loc[max_silhouette_idx,
    ['topics_count',
     'outliers_count',
     'n_neighbors']]

    axs[0].annotate(f"Neighbors: {max_silhouette_values['n_neighbors']}\n"
                    f"Topics: {max_silhouette_values['topics_count']}\n"
                    f"Outliers: {max_silhouette_values['outliers_count']}",
                    xy=(max_silhouette_values['topics_count'],
                        max_silhouette_values['outliers_count']),
                    xycoords='data',
                    xytext=(max_silhouette_values['topics_count'] - 60,
                            max_silhouette_values['outliers_count'] - 200),
                    fontsize=10, textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="arc3, rad=0.35",
                                    linewidth=1,
                                    edgecolor='k',
                                    linestyle='-')
                    )

    # Second subplot
    max_silhouette_idx = meta[meta['columns'] == 'column12345']['silhouette_score'].idxmax()
    max_silhouette_values = meta.loc[max_silhouette_idx, ['topics_count',
                                                          'outliers_count',
                                                          'n_neighbors']]

    axs[1].annotate(f"Neighbors: {max_silhouette_values['n_neighbors']}\n"
                    f"Topics: {max_silhouette_values['topics_count']}\n"
                    f"Outliers: {max_silhouette_values['outliers_count']}",
                    xy=(max_silhouette_values['topics_count'],
                        max_silhouette_values['outliers_count']),
                    xycoords='data',
                    xytext=(max_silhouette_values['topics_count'] + 50,
                            max_silhouette_values['outliers_count'] - 300),
                    fontsize=10, textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="arc3, rad=0.35",
                                    linewidth=1,
                                    edgecolor='k',
                                    linestyle='-')
                    )

    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(),
                             '..',
                             'figures',
                             'model124_v_12345.pdf'),
                bbox_inches='tight')