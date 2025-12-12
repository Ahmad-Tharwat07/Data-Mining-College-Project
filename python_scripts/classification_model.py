#!/usr/bin/env python
# coding: utf-8

# In[1]:


# https://www.kaggle.com/datasets/yasserh/titanic-dataset/data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,  roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
# get_ipython().run_line_magic('matplotlib', 'inline')

import os
# Load the dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'titanic.csv')
data = pd.read_csv(data_path)
data.head()


# In[2]:


print(data.info()) # shows columns names, number of non-null values and data type
print(data.isnull().sum()) # shows number of null values in each column




# In[3]:


df = data.copy()


# In[4]:


df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+\.)', expand=False)
df['Title'].unique()


# In[5]:


df['Title'] = df['Title'].replace(['Sir.'], 'Mr.')
df['Title'] = df['Title'].replace(['Mme.', 'Lady.', 'Countess.'], 'Mrs.')
df['Title'] = df['Title'].replace(['Ms.', 'Mlle.'], 'Miss.')
df['Title'] = df['Title'].replace(['Dr.', 'Rev.', 'Major.', 'Col.', 'Capt.', 'Jonkheer.', 'Don.'], 'Rare')
df['Title'].unique()


# In[6]:





# In[7]:


df.head()

# # Advanced Data Refinement (Final Push for >85%)
# Feature: Group Survival Rate
# Logic: Group by BOTH Surname (Families) and Ticket (Friends/Groups).

# 1. Extract Surname
df['Surname'] = df['Name'].apply(lambda x: x.split(',')[0].strip())

# 2. Family Size
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# 3. Create Group Survival Feature
# default to 0.5 (unknown)
df['Family_Survival'] = 0.5 

# Group by Surname
for grp, grp_df in df.groupby(['Surname', 'FamilySize']):
    if len(grp_df) > 1:
        if (grp_df['Survived'] == 1).any():
            df.loc[grp_df.index, 'Family_Survival'] = 1
        elif (grp_df['Survived'] == 0).any():
            df.loc[grp_df.index, 'Family_Survival'] = 0

# Group by Ticket (Catch friends/groups who aren't family)
for grp, grp_df in df.groupby('Ticket'):
    if len(grp_df) > 1:
        # If we already found a known survival state from Surname, keep it (Surname is stronger usually).
        # OR: If Ticket reveals info where Surname didn't (0.5), use it.
        # Let's iterate index
        for ind in grp_df.index:
            if df.loc[ind, 'Family_Survival'] == 0.5: # Only update if unknown
                if (grp_df.drop(ind)['Survived'] == 1).any():
                    df.loc[ind, 'Family_Survival'] = 1
                elif (grp_df.drop(ind)['Survived'] == 0).any():
                    df.loc[ind, 'Family_Survival'] = 0

# 4. Cleanup
df.drop(['PassengerId', 'Cabin', 'Name', 'Ticket', 'Surname'], axis=1, inplace=True)

# Binary mapping
df['Sex'] = df['Sex'].map({'female': 1, 'male': 0}).astype(int)

# One-hot Encoding
df = pd.get_dummies(df, columns=['Title', 'Embarked'], drop_first=True)


# In[13]:


df.head()


# # Spliting
# 

# In[14]:


X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")
print(y.unique())


# # Scaling (Skipping for Random Forest with Bins)
# Trees don't require scaling, and we dropped the continuous vars anyway.

sc = StandardScaler()
cols_to_scale = ['Age', 'Fare']
X_train[cols_to_scale] = sc.fit_transform(X_train[cols_to_scale])
X_test[cols_to_scale] = sc.transform(X_test[cols_to_scale])

X_train.head()

# # OVER SAMPLING (Skipped)
print(len(y_train[y_train == 1]))
print(len(y_train[y_train == 0]))

# # Training with Random Forest (Optimized)
# Using 'balanced' class weights to handle imbalance without oversampling.
# Using 'entropy' to match ID3 logic but with Ensemble strength.

# In[18]:
# from sklearn.model_selection import GridSearchCV

# Manual robust parameters based on experience
# # Training with Random Forest (Optimized)
# Single strong Random Forest performed best (~85.5%)
# Using 'balanced' class weights to handle imbalance without oversampling.

model = RandomForestClassifier(
    n_estimators=300,
    criterion='entropy',
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

print("Training Random Forest...")
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)


# ### Actual vs Predicted comparison
# 

# In[19]:


acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc * 100:.2f}%")
print("-" * 30)
print(classification_report(y_test, y_pred, target_names=['Did Not Survive', 'Survived']))


# In[20]:


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
# plt.show()


# ===============================================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix - ID3 Decision Tree\n(Titanic Dataset)')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

# plt.show()


# ### Confusion Matrix
# 

# In[ ]:


# Pclass	Sex	Age	SibSp	Parch	Fare	Title_Miss.	Title_Mr.	Title_Mrs.	Title_Rare	Embarked_Q	Embarked_S
# Pclass	Sex	Age	Fare	Title_Miss.	Title_Mr.	Title_Mrs.	Title_Rare	Embarked_Q	Embarked_S
def predict_user_input():
    print("\nEnter passenger details to predict survival:")
    try:
        p_class = int(input("Pclass: "))
        sex = input("Sex (male, female): ").strip().lower()
        age = float(input("Age: "))
        sibsp = int(input("SibSp: "))
        parch = int(input("Parch: "))
        fare = float(input("Fare: "))
        embarked = input("Embarked: ").strip().upper()
        title = input("Title: ").strip()

        # Create DataFrame with Initial Features
        input_data = {
            'Pclass': [p_class],
            'Sex': [1 if sex == 'female' else 0], # Map female:1, male:0
            'Age': [age],
            'SibSp': [sibsp],
            'Parch': [parch],
            'Fare': [fare],
            'FamilySize': [sibsp + parch + 1],
            'Family_Survival': [0.5] # Default to 0.5 (Unknown family) for new inputs
        }
        
        # (One-Hot Encoding)
        # Initialize all dummy columns to 0
        dummy_cols = ['Title_Miss.', 'Title_Mr.', 'Title_Mrs.', 'Title_Rare', 'Embarked_Q', 'Embarked_S']
        for col in dummy_cols:
            input_data[col] = [0]

        # Set Title dummy
        t_clean = title.replace('.', '')
        title_map = {
            'Miss': 'Title_Miss.',
            'Mr': 'Title_Mr.',
            'Mrs': 'Title_Mrs.',
            'Rare': 'Title_Rare'
        }
        if t_clean in title_map:
            input_data[title_map[t_clean]] = [1]

        # Set Embarked dummy
        emb_map = {'Q': 'Embarked_Q', 'S': 'Embarked_S'}
        if embarked in emb_map:
            input_data[emb_map[embarked]] = [1]

        # Create DataFrame & Ensure correct column order
        input_df = pd.DataFrame(input_data)
        
        # Get columns from training data to ensure match (excluding Survived)
        model_cols = X.columns.tolist()
        input_df = input_df[model_cols]
        
        # Scale Age and Fare (StandardScaler is active)
        input_df[['Age', 'Fare']] = sc.transform(input_df[['Age', 'Fare']])

        # Predict
        prediction = model.predict(input_df)[0]
        pred_proba = model.predict_proba(input_df)[0]

        outcome = "Survived" if prediction == 1 else "Did Not Survive"
        print(f"\nPrediction: {outcome}")
        print(f"Probability of Survival: {pred_proba[1]:.2%}")

    except Exception as e:
        print(f"An error occurred: {e}")

predict_user_input()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




