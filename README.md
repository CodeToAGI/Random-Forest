# CodeToAGI — Machine Learning From Scratch
**Episode 82: Random Forest Explained — Random Forest vs Decision Tree**

---

## 🎯 Challenge Task (EP82)

**Task Recap:**

Use the same 10-email spam/ham dataset from previous episodes.

1. Train a `RandomForestClassifier(n_estimators=100)` and compare accuracy with a single `DecisionTreeClassifier`
2. Compare `feature_importances_` between the single tree and the forest
3. **Bonus**: Test different `n_estimators` values (1, 5, 10, 50, 100) and observe the effect

---

## 📁 Files

- **`ep82_challenge_solution.py`** — Complete solution with detailed output
- `README.md` — This file

---

## 🧪 Expected Results

- Both models achieve **100% accuracy** on this clean dataset
- `caps_ratio` dominates feature importance (~0.89) in both models
- Increasing `n_estimators` gives more stable predictions on noisier data

---

## 🚀 How to Run

```bash
# Activate environment
myenv\Scripts\activate

# Run the solution
python ep82_challenge_solution.py
