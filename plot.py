import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm

# Load data
df = pd.read_csv('round_table_data.csv', index_col=0)
speed_range = [df.index, df.columns]

# Create the plot
fig, ax = plt.subplots()

# Create seaborn heatmap
boundaries = np.arange(df.min(None) - 0.5, df.max(None) + 1.5)
norm = BoundaryNorm(boundaries=boundaries, ncolors=256)
sns.heatmap(
    df,
    cmap='viridis',
    cbar_kws={'label': 'Turns', 'ticks': boundaries + 0.5},
    ax=ax,
    xticklabels=10,  # Show every 10th x-tick
    yticklabels=10,  # Show every 10th y-tick
    norm=norm,
)

ax.set_xlabel("Firefly's speed")
ax.set_ylabel("Bronya's speed")
ax.set_title('Turns heatmap')

# Invert y-axis to match your original (smallest speeds at bottom)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("plot.png", dpi=300, format="png")
plt.show()
