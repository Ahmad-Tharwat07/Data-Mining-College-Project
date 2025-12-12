# Decision Tree Classification (Project Baseline)

## 1. The Algorithm: Decision Tree (ID3)

The initial model used for this project was a **Decision Tree Classifier** based on the **ID3 (Iterative Dichotomiser 3)** logic.

### How it Works (The Flowchart Logic)

Think of this model as a series of Yes/No questions designed to split the passengers into "Survived" or "Died" buckets.

1.  **Root Node**: Is `Sex` male or female?
    - _Female_ → High chance of survival.
    - _Male_ → Low chance of survival.
2.  **Next Split**: Is `Pclass` < 3?
    - _Rich_ → Better chance.
    - _Poor_ → Worse chance.
3.  **Leaf Node**: The final bucket determines the prediction (0 or 1).

### Technical Configuration

- **Library**: `sklearn.tree.DecisionTreeClassifier`
- **Criterion**: `entropy` (Information Gain).
- **Max Depth**: `5` (Restricted to prevent memorizing the data).

## 2. Performance & Limitations

While easy to understand, the Decision Tree hit a "Performance Ceiling" at approximately **80-84% accuracy**.

### Why it Failed to Reach 85%+

1.  **High Variance**: Single trees are unstable. Changing just a few rows of training data can completely change the structure of the tree.
2.  **Overfitting vs. Underfitting**:
    - If the tree is too deep, it memorizes the training data (Overfitting).
    - If the tree is too shallow, it misses complex patterns (Underfitting).
3.  **The "Independence" Flaw**: The Decision Tree treats every passenger as an independent event. It fails to recognize that passengers (families/groups) influenced each other's survival.

## 3. Conclusion

The Decision Tree served as a good baseline but lacked the robustness required for high-accuracy predictions on this small, noisy dataset. This necessitated the shift to an **Ensemble Method** (Random Forest).
