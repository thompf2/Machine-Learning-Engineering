import os
import pandas as pd
import numpy as np
from google.colab import drive
import xgboost as xgb
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# mount google drive
drive.mount('/content/drive')

# load and preprocess data
dataset_dir = "/content/drive/MyDrive/Machine Learning Engineering/gt"

# function to load each file efficiently
def process_file(file_path):
    data = np.genfromtxt(file_path, delimiter=" ")
    return pd.DataFrame({
        "x_coords": data[:, 0].astype("float32"),  # Reduce to float32
        "y_coords": data[:, 1].astype("float32"),
        "z_coords": data[:, 2].astype("float32"),
        "defect_label": data[:, 3].astype("int8")  # Reduce to int8
    })

# load and process files one by one
df_list = []
for filename in os.listdir(dataset_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(dataset_dir, filename)
        df_list.append(process_file(file_path))

# combine all into a single DataFrame
df = pd.concat(df_list, ignore_index=True)

# add spatial features
df["distance_from_origin"] = np.sqrt(df["x_coords"]**2 + df["y_coords"]**2 + df["z_coords"]**2)

# compute curvature approximation (local variation in Z)
df["curvature"] = abs(df["z_coords"] - df["z_coords"].rolling(5, center=True, min_periods=1).mean())

# prepare data with new features
features = ["x_coords", "y_coords", "z_coords", "distance_from_origin", "curvature"]
X = df[features]
y = df["defect_label"]
