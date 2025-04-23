y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# find the best threshold using a weighted F-beta score (beta=2 favors recall more)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
fbeta_scores = (1 + 2**2) * (precision * recall) / ((2**2 * precision) + recall)
best_threshold = thresholds[np.argmax(fbeta_scores[:-1])]  # exclude last precision/recall pair

print(f"Optimized Threshold (Weighted Recall): {best_threshold:.4f}")

y_pred = (y_pred_proba > best_threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Defect", "Defect"], yticklabels=["No Defect", "Defect"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# feature importance visualization
feature_importance = best_model.feature_importances_
plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importance, y=features)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("XGBoost Feature Importance")
plt.show()

# extract values from confusion matrix
TN, FP, FN, TP = conf_matrix.ravel()

# calculate rates
TrueDefectRate = TP / (TP + FN)
FalseDefectRate = FP / (FP + TN)

TrueNonDefectRate = TN / (TN + FP)
FalseNonDefectRate = FN / (FN + TP)

print(f"True Defect Rate: {TrueDefectRate:.4f}")
print(f"False Defect Rate: {FalseDefectRate:.4f}")
print(f"True Non-Defect Rate: {TrueNonDefectRate:.4f}")
print(f"False Non-Defect Rate: {FalseNonDefectRate:.4f}")

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

