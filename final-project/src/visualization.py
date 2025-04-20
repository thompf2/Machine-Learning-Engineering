from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# add labels to test set for visualization
viz_df = X_test.copy()
viz_df["True Label"] = y_test.values
viz_df["Predicted Label"] = y_pred

# side by side scatter plot
fig = plt.figure(figsize=(16, 7))

# true labels plot
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
colors_true = ['blue' if label == 0 else 'red' for label in viz_df["True Label"]]
ax1.scatter(viz_df["x_coords"], viz_df["y_coords"], viz_df["z_coords"], c=colors_true, alpha=0.6)
ax1.set_title("True Defect Labels")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

# predicted labels plot
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
colors_pred = ['blue' if label == 0 else 'red' for label in viz_df["Predicted Label"]]
ax2.scatter(viz_df["x_coords"], viz_df["y_coords"], viz_df["z_coords"], c=colors_pred, alpha=0.6)
ax2.set_title("Predicted Defect Labels")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")

plt.suptitle("3D Scatter: True vs Predicted Labels", fontsize=16)
plt.tight_layout()
plt.show()

