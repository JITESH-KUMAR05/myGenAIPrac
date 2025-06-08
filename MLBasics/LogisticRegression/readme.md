# Logistic Regression with Hyperparameter Tuning

## GridSearchCV Explanation

### What is GridSearchCV?
GridSearchCV is a technique used to find the best combination of hyperparameters for a machine learning model by systematically testing all possible combinations from a predefined parameter grid.

### Code Breakdown:
```python
grid = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=cv, n_jobs=-1)
```

#### Parameters Explained:

1. **estimator=model**: 
   - The machine learning algorithm to optimize (LogisticRegression in this case)
   - This is the base model that will be trained with different parameter combinations

2. **param_grid=params**: 
   - Dictionary containing hyperparameters and their possible values to test
   - Example: `{'penalty': ['l1', 'l2'], 'C': [0.1, 1.0, 10], 'solver': ['liblinear', 'lbfgs']}`
   - GridSearchCV will test ALL combinations: 2 × 3 × 2 = 12 different models

3. **scoring='accuracy'**: 
   - Metric used to evaluate model performance
   - Other options: 'precision', 'recall', 'f1', 'roc_auc'
   - GridSearchCV will select the combination with highest accuracy

4. **cv=cv**: 
   - Cross-validation strategy
   - Determines how data is split for validation during hyperparameter search
   - Each parameter combination is evaluated using this CV strategy

5. **n_jobs=-1**: 
   - Number of CPU cores to use for parallel processing
   - -1 means use all available cores
   - Speeds up the search process significantly

### Example Workflow:
```python
# Define parameter grid
params = {
    'penalty': ['l1', 'l2'],
    'C': [0.1, 1.0, 10],
    'solver': ['liblinear', 'lbfgs']
}

# GridSearchCV will test these combinations:
# 1. penalty='l1', C=0.1, solver='liblinear'
# 2. penalty='l1', C=0.1, solver='lbfgs'
# 3. penalty='l1', C=1.0, solver='liblinear'
# ... and so on (12 total combinations)
```

## Cross-Validation: StratifiedKFold vs Integer

### Question: Can we use `cv=5` instead of `cv=StratifiedKFold()`?

**Answer: YES!** You can absolutely use `cv=5` directly.

#### Differences:

**Using `cv=5`:**
```python
grid = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=5, n_jobs=-1)
```
- Uses regular KFold cross-validation
- Splits data into 5 folds randomly
- May not preserve class distribution in each fold

**Using `cv=StratifiedKFold()`:**
```python
cv = StratifiedKFold(n_splits=5)  # default is 5 splits
grid = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=cv, n_jobs=-1)
```
- Ensures each fold maintains the same proportion of samples from each class
- Better for imbalanced datasets
- More reliable performance estimates

#### When to use which:

- **Use `cv=5`**: For balanced datasets or when simplicity is preferred
- **Use `cv=StratifiedKFold()`**: For imbalanced datasets or when you want more control over the CV process

#### Example of class distribution preservation:

```python
# Original dataset: 70% class 0, 30% class 1

# Regular KFold (cv=5):
# Fold 1: 60% class 0, 40% class 1  ❌ Distribution changed
# Fold 2: 80% class 0, 20% class 1  ❌ Distribution changed

# StratifiedKFold:
# Fold 1: 70% class 0, 30% class 1  ✅ Distribution preserved
# Fold 2: 70% class 0, 30% class 1  ✅ Distribution preserved
```

### Recommendation:
For most classification problems, especially with imbalanced classes, use `StratifiedKFold()` for more reliable results.