# K-Nearest Neighbors Classification on Iris Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("K-NEAREST NEIGHBORS CLASSIFICATION - IRIS DATASET")
print("=" * 70)

# Step 1: Load and Explore Data
print("\n1. LOADING AND EXPLORING DATA")
print("-" * 40)

# Load the dataset: prefer local CSV, fall back to sklearn's bundled iris dataset
try:
    df_iris = pd.read_csv('Iris.csv')
    print("Loaded dataset from 'Iris.csv'.")
except FileNotFoundError:
    from sklearn import datasets as _datasets
    iris_raw = _datasets.load_iris()
    # create DataFrame with the same column names expected by the script
    df_iris = pd.DataFrame(iris_raw.data, columns=[
        'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'
    ])
    # add Species names and a dummy Id column to match Kaggle-style CSVs used elsewhere
    df_iris['Species'] = pd.Categorical.from_codes(iris_raw.target, iris_raw.target_names)
    df_iris.insert(0, 'Id', range(1, len(df_iris) + 1))
    print("Iris.csv not found — loaded dataset from sklearn.datasets.load_iris().")

print(f"Dataset Shape: {df_iris.shape}")
print(f"\nFirst 5 rows:")
print(df_iris.head())

print(f"\nDataset Info:")
print(df_iris.info())
print(f"\nMissing Values:\n{df_iris.isnull().sum()}")

print(f"\nSpecies Distribution:")
print(df_iris['Species'].value_counts())

# Step 2: Data Preprocessing
print("\n2. DATA PREPROCESSING")
print("-" * 40)

# Create a copy for preprocessing
df_processed = df_iris.copy()

# Select features and target
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = df_processed[features]
y = df_processed['Species']

print(f"Selected features: {features}")
print(f"Target variable: Species")
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_

print(f"\nEncoded classes: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# Step 3: Split Data
print("\n3. TRAIN-TEST SPLIT")
print("-" * 40)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")
print(f"Class distribution in training set: {np.bincount(y_train)}")
print(f"Class distribution in testing set: {np.bincount(y_test)}")

# Step 4: Feature Scaling (Crucial for KNN!)
print("\n4. FEATURE SCALING")
print("-" * 40)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed!")
print(f"Scaled training data - Mean: {X_train_scaled.mean():.3f}, Std: {X_train_scaled.std():.3f}")

# Step 5: Train KNN Model
print("\n5. TRAINING KNN MODEL")
print("-" * 40)

# Create and train KNN classifier
knn_model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_model.fit(X_train_scaled, y_train)

print("KNN model training completed!")
print(f"Number of neighbors (k): {knn_model.n_neighbors}")
print(f"Distance metric: {knn_model.metric}")

# Step 6: Make Predictions
print("\n6. MAKING PREDICTIONS")
print("-" * 40)

y_pred = knn_model.predict(X_test_scaled)
y_pred_proba = knn_model.predict_proba(X_test_scaled)

print("Predictions completed!")
print(f"First 10 predictions: {y_pred[:10]}")
print(f"First 10 actual values: {y_test[:10]}")

# Step 7: Evaluate Model
print("\n7. MODEL EVALUATION")
print("-" * 40)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title('Confusion Matrix - KNN\n(Iris Dataset)')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('knn_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Step 8: Find Optimal K Value
print("\n8. FINDING OPTIMAL K VALUE")
print("-" * 40)

# Test different k values
k_range = range(1, 21)
accuracies = []

for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_scaled, y_train)
    y_pred_temp = knn_temp.predict(X_test_scaled)
    accuracies.append(accuracy_score(y_test, y_pred_temp))

# Plot accuracy vs k
plt.figure(figsize=(10, 6))
plt.plot(k_range, accuracies, marker='o', linestyle='-', color='teal')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy Score')
plt.title('KNN Accuracy vs Number of Neighbors\n(Iris Dataset)')
plt.xticks(k_range)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('knn_accuracy_vs_k.png', dpi=300, bbox_inches='tight')
plt.show()

best_k = k_range[np.argmax(accuracies)]
best_accuracy = max(accuracies)
print(f"Optimal k value: {best_k} with accuracy: {best_accuracy:.4f}")

# Step 9: Visualizations
print("\n9. DATA VISUALIZATIONS")
print("-" * 40)

# 9.1 Pairplot of features
plt.figure(figsize=(12, 10))
df_plot = df_iris.copy()
sns.pairplot(df_plot.drop('Id', axis=1), hue='Species', palette='Dark2', diag_kind='hist')
plt.suptitle('Iris Dataset Pairplot - Feature Relationships', y=1.02)
plt.savefig('knn_iris_pairplot.png', dpi=300, bbox_inches='tight')
plt.show()

# 9.2 Actual vs Predicted comparison
comparison_df = pd.DataFrame({
    'Actual': label_encoder.inverse_transform(y_test),
    'Predicted': label_encoder.inverse_transform(y_pred),
    'Correct': y_test == y_pred
})

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
comparison_df['Actual'].value_counts().plot(kind='bar', color=['lightcoral', 'lightgreen', 'lightblue'])
plt.title('Actual Species Distribution\n(Test Set)')
plt.xlabel('Species')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
comparison_df['Predicted'].value_counts().plot(kind='bar', color=['coral', 'green', 'blue'])
plt.title('Predicted Species Distribution\n(Test Set)')
plt.xlabel('Species')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('knn_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()

# 9.3 Feature correlation heatmap
plt.figure(figsize=(8, 6))
correlation_matrix = df_iris[features + ['Species']].copy()
correlation_matrix['Species'] = label_encoder.transform(correlation_matrix['Species'])
sns.heatmap(correlation_matrix.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap\n(Iris Dataset)')
plt.tight_layout()
plt.savefig('knn_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Step 10: Predict New Values
print("\n10. PREDICTING NEW VALUES")
print("-" * 40)

# Create new flower measurements
new_flowers = pd.DataFrame({
    'SepalLengthCm': [5.1, 6.7, 4.8],
    'SepalWidthCm': [3.5, 3.1, 2.9],
    'PetalLengthCm': [1.4, 5.6, 4.3],
    'PetalWidthCm': [0.2, 2.4, 1.3]
})

print("New flower measurements for prediction:")
print(new_flowers)

# Scale new data
new_flowers_scaled = scaler.transform(new_flowers)

# Make predictions
new_predictions = knn_model.predict(new_flowers_scaled)
new_probabilities = knn_model.predict_proba(new_flowers_scaled)

print("\nPrediction Results for New Flowers:")
for i, (pred, prob) in enumerate(zip(new_predictions, new_probabilities)):
    species_name = label_encoder.inverse_transform([pred])[0]
    confidence = prob[pred]
    print(f"Flower {i+1}: {species_name} (confidence: {confidence:.3f})")

# Step 11: Manual KNN Implementation
print("\n11. MANUAL KNN IMPLEMENTATION")
print("-" * 40)

class SimpleKNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        # Calculate Euclidean distances
        distances = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.X_train]
        
        # Get k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Test manual implementation on 2 features for simplicity
X_simple = df_iris[['PetalLengthCm', 'PetalWidthCm']].values
y_simple = label_encoder.transform(df_iris['Species'])

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y_simple, test_size=0.2, random_state=42
)

# Scale the simple dataset
scaler_simple = StandardScaler()
X_train_s_scaled = scaler_simple.fit_transform(X_train_s)
X_test_s_scaled = scaler_simple.transform(X_test_s)

# Use manual KNN
manual_knn = SimpleKNN(k=5)
manual_knn.fit(X_train_s_scaled, y_train_s)
y_pred_manual = manual_knn.predict(X_test_s_scaled)

manual_accuracy = accuracy_score(y_test_s, y_pred_manual)
print(f"Manual KNN Implementation Accuracy: {manual_accuracy:.4f}")

# Step 12: Model Summary
print("\n" + "=" * 70)
print("MODEL SUMMARY - K-NEAREST NEIGHBORS")
print("=" * 70)
print(f"Dataset: Iris Species Classification")
print(f"Features used: {len(features)}")
print(f"Final Accuracy: {accuracy:.4f}")
print(f"Optimal k value: {best_k}")
print(f"Distance metric: Euclidean")
print(f"Manual implementation accuracy: {manual_accuracy:.4f}")

print("\nKey Insights:")
print("- KNN performs exceptionally well on the Iris dataset")
print("- Feature scaling is crucial for KNN's performance")
print("- The optimal k value can be found through experimentation")
print("- Excellent for datasets with clear clusters and numerical features")

print("\nFiles saved:")
print("- knn_confusion_matrix.png")
print("- knn_accuracy_vs_k.png")
print("- knn_iris_pairplot.png")
print("- knn_actual_vs_predicted.png")
print("- knn_correlation_heatmap.png")

print("\n" + "=" * 70)
print("COMPARISON WITH DECISION TREE:")
print("=" * 70)
print("✓ KNN achieved near-perfect accuracy on Iris dataset")
print("✓ Decision trees are more interpretable but may have lower accuracy")
print("✓ KNN requires feature scaling, Decision Trees don't")
print("✓ Choice depends on dataset characteristics and project requirements")