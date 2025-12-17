"""
Demo script to show the improved confusion matrix visualization
"""
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

# Create sample confusion matrix similar to the one in the user's image
labels = [
    'Hard coral',
    'Bare substrate',
    'Rubble',
    'Coral::Encrusting',
    'Hard coral',
    'Sand',
    'Soft coral',
    'Dead coral',
    'Sargassum',
    'Coral::Digitate'
]

# Create a matrix with similar pattern to the user's image
# Mostly zeros with some diagonal values and a few off-diagonal
matrix = np.array([
    [0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [0,   0,  50,   0,   0,  50,   0,   0,   0,   0],
    [0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [0,   0,   0,   0,  75,  25,   0,   0,   0,   0],
    [0,   0,   0,   0,   0,  83,   0,  16,   0,   0],
    [0,   0,   0,   0,   0, 100,   0,   0,   0,   0],
    [0,   0,   0,   0,   0, 100,   0,   0,   0,   0],
    [0,   0,   0,   0,   0,   0,   0,   0, 100,   0],
    [0,   0,   0,   0,   0,   0,   0,   0,   0, 100],
])

# Create figure with size scaled to number of labels
num_labels = len(labels)
fig_size = max(12, num_labels * 0.6)
fig, ax = plt.subplots(figsize=(fig_size, fig_size))

# Create confusion matrix display
display = sklearn.metrics.ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=labels
)

# Plot with format string to avoid scientific notation
display.plot(
    ax=ax,
    cmap='viridis',
    values_format='.0f'  # Display as integers without scientific notation
)

# Rotate x-axis labels to prevent overlap
plt.setp(
    ax.get_xticklabels(),
    rotation=45,
    ha='right',
    rotation_mode='anchor',
    fontsize=max(8, min(12, 150 / num_labels))
)

# Adjust y-axis label font size for consistency
plt.setp(
    ax.get_yticklabels(),
    fontsize=max(8, min(12, 150 / num_labels))
)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save figure
plt.savefig('confusion_matrix_demo.png', dpi=150, bbox_inches='tight')
print("Saved confusion matrix to: confusion_matrix_demo.png")

plt.close(fig)
