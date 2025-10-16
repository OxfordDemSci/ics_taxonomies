import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable


def main():
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5),
                            sharex=True)

    meta = pd.read_csv(os.path.join(os.getcwd(),
                                    '..',
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
        ax.set_xlim(60,200)

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
        "cols = ['1. Summary of the impact',\n"+
        "            '2. Underpinning research',\n"+
        "            '3. References to the research',\n"+
        "            '4. Details of the impact',\n"+
        "            '5. Sources to corroborate the impact']",
        prop=dict(size=8), frameon=True, loc='upper right')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.4")
    axs[1].add_artist(at)

    at = AnchoredText(
        "cols = ['1. Summary of the impact',\n"+
        "            '2. Underpinning research',\n"+
        "            '4. Details of the impact']",
        prop=dict(size=8), frameon=True, loc='upper right')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.4")
    axs[0].add_artist(at)

    # First subplot
    max_silhouette_idx = meta[meta['columns']=='columns124']['silhouette_score'].idxmax()
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
                    xytext=(max_silhouette_values['topics_count']-60,
                            max_silhouette_values['outliers_count']-200),
                    fontsize=10, textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="arc3, rad=0.35",
                                    linewidth=1,
                                    edgecolor='k',
                                    linestyle='-')
                   )

    # Second subplot
    max_silhouette_idx = meta[meta['columns']=='column12345']['silhouette_score'].idxmax()
    max_silhouette_values = meta.loc[max_silhouette_idx, ['topics_count',
                                                          'outliers_count',
                                                          'n_neighbors']]

    axs[1].annotate(f"Neighbors: {max_silhouette_values['n_neighbors']}\n"
                    f"Topics: {max_silhouette_values['topics_count']}\n"
                    f"Outliers: {max_silhouette_values['outliers_count']}",
                    xy=(max_silhouette_values['topics_count'],
                        max_silhouette_values['outliers_count']),
                    xycoords='data',
                    xytext=(max_silhouette_values['topics_count']+50,
                            max_silhouette_values['outliers_count']-300),
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
                             '..',
                             'outputs',
                             'figures',
                             'model124_v_12345.pdf'),
                             bbox_inches='tight')

if __name__ == "__main__":
    main()