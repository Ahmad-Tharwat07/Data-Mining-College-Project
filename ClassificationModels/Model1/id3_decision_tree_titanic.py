# ID3 Decision Tree Classification on Titanic Dataset
# https://www.kaggle.com/datasets/yasserh/titanic-dataset/data
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ID3 DECISION TREE CLASSIFICATION - TITANIC DATASET")
print("=" * 70)

# Step 1: Load and Explore Data
# Load the dataset
try:
    df_titanic = pd.read_csv('titanic.csv')
    print("Loaded dataset from 'titanic.csv'.")
except FileNotFoundError:
    print("Not found")
    
print(f"Dataset Shape: {df_titanic.shape}")
print(f"\nFirst 5 rows:")
print(df_titanic.head(5))

print(f"\nDataset Info:")
print(df_titanic.info()) # shows columns names, number of non-null values and data type
print(f"\nMissing Values:\n{df_titanic.isnull().sum()}") # shows number of null values in each column

# Step 2: Data Preprocessing
print("\n2. DATA PREPROCESSING")
print("-" * 40)


######################################## 1. Handle missing values ########################################
df_processed = df_titanic.copy()

imputer_age = SimpleImputer(strategy='median') # cus mean is sensitive to outliers.
df_processed[['Age']] = imputer_age.fit_transform(df_processed[['Age']])

imputer_embarked = SimpleImputer(strategy='most_frequent')
df_processed[['Embarked']] = imputer_embarked.fit_transform(df_processed[['Embarked']])

df_processed.drop('Cabin', axis=1, inplace=True) # Malo4 lazma + most of it is null

# helper 
# print("Mode for Embarked:", imputer_embarked.statistics_[0])
# print("Median for Age:", imputer_age.statistics_[0])


# Select relevant features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df_processed[features] # Processed data
y = df_processed['Survived'] # Predition

# print(f"Selected features: {features}")
# print(f"Target variable: Survived")
# print(f"Features shape: {X.shape}")
# print(f"Target shape: {y.shape}")

############################################### 2.ENCODING ###############################################
label_encoder_sex = LabelEncoder()
label_encoder_embarked = LabelEncoder()

X_encoded = X.copy()
X_encoded['Sex'] = label_encoder_sex.fit_transform(X['Sex'])
X_encoded['Embarked'] = label_encoder_embarked.fit_transform(X['Embarked'])

print(f"\nAfter encoding categorical variables:")
print(X_encoded.head())

############################################### 3.SPLITING ################################################
print("\n3. TRAIN-TEST SPLIT")
print("-" * 40)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")
print(f"Training survival rate: {y_train.mean():.3f}")
print(f"Testing survival rate: {y_test.mean():.3f}")

############################################## 4. SCALING #################################################
sc = StandardScaler()
X_train[['Age', 'Fare']] = sc.fit_transform(X_train[['Age', 'Fare']])
X_test[['Age', 'Fare']] = sc.transform(X_test[['Age', 'Fare']])

print(f"\nAfter scaling numerical variables:")
print(X_train.head())
############################################## 5. OVER SAMPLING ###########################################

ros = RandomOverSampler()
X_train, y_train = ros.fit_resample(X_train, y_train)

print(len(y_train[y_train == 1]))
print(len(y_train[y_train == 0]))

########################################## 6.TRAINING & VISUALS ###########################################
print("\n4. TRAINING ID3 DECISION TREE MODEL")
print("-" * 40)

# Create and train Decision Tree with entropy (ID3 algorithm)
id3_model = DecisionTreeClassifier(
    criterion='entropy',  # This makes it ID3-like
    random_state=42,
    max_depth=5  # Limit depth for better visualization
)
id3_model.fit(X_train, y_train)

print("Model training completed!")
print(f"Number of nodes in the tree: {id3_model.tree_.node_count}")

# Step 5: Make Predictions
print("\n5. MAKING PREDICTIONS")
print("-" * 40)

y_pred = id3_model.predict(X_test)
y_pred_proba = id3_model.predict_proba(X_test)

print("Predictions completed!")
print(f"First 10 predictions: {y_pred[:10]}")
print(f"First 10 actual values: {y_test.values[:10]}")

# Step 6: Evaluate Model
print("\n6. MODEL EVALUATION")
print("-" * 40)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Survived', 'Survived'], 
            yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix - ID3 Decision Tree\n(Titanic Dataset)')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('id3_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Step 7: Visualizations
print("\n7. DATA VISUALIZATIONS")
print("-" * 40)

# 7.1 Feature Importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': id3_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
plt.title('Feature Importance - ID3 Decision Tree\n(Titanic Dataset)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('id3_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nFeature Importance Ranking:")
for i, row in feature_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# 7.2 Decision Tree Visualization
plt.figure(figsize=(20, 10))
plot_tree(id3_model, 
          feature_names=features, 
          class_names=['Not Survived', 'Survived'],
          filled=True, 
          rounded=True, 
          fontsize=10,
          proportion=True)
plt.title('ID3 Decision Tree Visualization - Titanic Survival Prediction')
plt.tight_layout()
plt.savefig('id3_decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()

# 7.3 Actual vs Predicted comparison
comparison_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred,
    'Correct': y_test.values == y_pred
})

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
comparison_df['Actual'].value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
plt.title('Actual Survival Distribution\n(Test Set)')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'], rotation=0)

plt.subplot(1, 2, 2)
comparison_df['Predicted'].value_counts().plot(kind='bar', color=['lightblue', 'lightpink'])
plt.title('Predicted Survival Distribution\n(Test Set)')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'], rotation=0)

plt.tight_layout()
plt.savefig('id3_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()

# Step 8: Predict New Values
print("\n8. PREDICTING NEW VALUES")
print("-" * 40)

# Create some new hypothetical passenger data
new_passengers = pd.DataFrame({
    'Pclass': [1, 3, 2],
    'Sex': ['female', 'male', 'female'],
    'Age': [25, 30, 45],
    'SibSp': [0, 1, 0],
    'Parch': [0, 0, 2],
    'Fare': [100, 20, 50],
    'Embarked': ['C', 'S', 'Q']
})

print("New passenger data for prediction:")
print(new_passengers)

# Preprocess new data
new_passengers_encoded = new_passengers.copy()
new_passengers_encoded['Sex'] = label_encoder_sex.transform(new_passengers['Sex'])
new_passengers_encoded['Embarked'] = label_encoder_embarked.transform(new_passengers['Embarked'])

# Make predictions
new_predictions = id3_model.predict(new_passengers_encoded)
new_probabilities = id3_model.predict_proba(new_passengers_encoded)

print("\nPrediction Results for New Passengers:")
for i, (pred, prob) in enumerate(zip(new_predictions, new_probabilities)):
    survival_status = "Survived" if pred == 1 else "Not Survived"
    survival_prob = prob[1] if pred == 1 else prob[0]
    print(f"Passenger {i+1}: {survival_status} (confidence: {survival_prob:.3f})")

# Step 9: Model Summary
print("\n" + "=" * 70)
print("MODEL SUMMARY - ID3 DECISION TREE")
print("=" * 70)
print(f"Dataset: Titanic Survival Prediction")
print(f"Features used: {len(features)}")
print(f"Final Accuracy: {accuracy:.4f}")
print(f"Most important feature: {feature_importance.iloc[0]['feature']}")
print(f"Tree Depth: {id3_model.get_depth()}")
print(f"Number of Leaves: {id3_model.get_n_leaves()}")

print("\nKey Insights:")
print("- Decision Trees provide excellent interpretability")
print("- Feature importance shows which factors most affect survival")
print("- The model can be easily visualized and understood")
print("- Good for datasets with mixed categorical and numerical features")

print("\nFiles saved:")
print("- id3_confusion_matrix.png")
print("- id3_feature_importance.png")
print("- id3_decision_tree.png")
print("- id3_actual_vs_predicted.png")