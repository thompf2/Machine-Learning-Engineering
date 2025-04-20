# confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Defect", "Defect"], yticklabels=["No Defect", "Defect"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

plt.savefig("plots/confusion_matrix.png")

# feature importance visualization
feature_importance = best_model.feature_importances_
plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importance, y=features)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("XGBoost Feature Importance")
plt.show()

plt.savefig("plots/feature_importance.png")

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

