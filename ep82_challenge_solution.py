"""
CodeToAGI — Machine Learning From Scratch
Episode 9: Random Forest Explained — Random Forest vs Decision Tree
Challenge Task — SOLUTION

Task recap (shown in the video):
  Same 10-email dataset as Episodes 7 & 8, each with two features:
    word_count  — how many words in the email
    caps_ratio  — fraction of words in ALL CAPS

  Email dataset:
    word_count  caps_ratio  label
    45          0.02        ham
    210         0.41        spam
    32          0.01        ham
    185         0.37        spam
    60          0.05        ham
    220         0.44        spam
    28          0.01        ham
    198         0.39        spam
    210         0.40        spam   ← test email 1
    38          0.02        ham    ← test email 2

  1) Fit RandomForestClassifier with n_estimators=100, compare accuracy
     to a single DecisionTreeClassifier (max_depth=None) on the 2 test emails.
  2) Print feature_importances_ for the Random Forest — does the forest
     agree with the single tree about which feature matters more?
  3) Bonus: vary n_estimators (1, 5, 10, 50, 100) and print accuracy
     for each — does more trees always help on this tiny dataset?

If you solved it differently and got the same story, that's valid.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ── Dataset ──────────────────────────────────────────────────────────────────
X_train = np.array([
    [45,  0.02],
    [210, 0.41],
    [32,  0.01],
    [185, 0.37],
    [60,  0.05],
    [220, 0.44],
    [28,  0.01],
    [198, 0.39],
], dtype=float)
y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1])   # 0=ham, 1=spam

X_test = np.array([
    [210, 0.40],   # email #9  — should be spam
    [38,  0.02],   # email #10 — should be ham
], dtype=float)
y_test = np.array([1, 0])

feature_names = ["word_count", "caps_ratio"]
class_names = ["ham", "spam"]

# ── Task 1: Single Decision Tree baseline ─────────────────────────────────────
single_tree = DecisionTreeClassifier(random_state=0)
single_tree.fit(X_train, y_train)
preds_tree = single_tree.predict(X_test)
acc_tree = (preds_tree == y_test).mean() * 100

print("── Task 1a: Single DecisionTree (max_depth=None) ──────────────────────")
print(f"  Test predictions : {['SPAM' if p else 'HAM' for p in preds_tree]}")
print(f"  Test accuracy    : {acc_tree:.0f}%\n")

# ── Task 1: Random Forest with 100 trees ──────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
preds_rf = rf.predict(X_test)
acc_rf = (preds_rf == y_test).mean() * 100

print("── Task 1b: RandomForestClassifier (n_estimators=100) ─────────────────")
print(f"  Number of trees  : {rf.n_estimators}")
print(f"  Test predictions : {['SPAM' if p else 'HAM' for p in preds_rf]}")
print(f"  Test accuracy    : {acc_rf:.0f}%\n")

print("── Comparison ─────────────────────────────────────────────────────────")
print(f"  Single Decision Tree accuracy : {acc_tree:.0f}%")
print(f"  Random Forest accuracy        : {acc_rf:.0f}%")
if acc_rf >= acc_tree:
    print("  The forest matches or beats the single tree — as expected on clean data.")
else:
    print("  Rare: single tree edged out the forest on this tiny dataset.")
print()

# ── Task 2: Feature importances — forest vs single tree ───────────────────────
print("── Task 2: Feature importances comparison ─────────────────────────────")
print("  Single tree feature_importances_:")
for name, imp in zip(feature_names, single_tree.feature_importances_):
    print(f"    {name:<12}: {imp:.3f}")
top_tree = feature_names[np.argmax(single_tree.feature_importances_)]
print(f"  → Most important (single tree): {top_tree}\n")

print("  Random Forest feature_importances_:")
for name, imp in zip(feature_names, rf.feature_importances_):
    print(f"    {name:<12}: {imp:.3f}")
top_rf = feature_names[np.argmax(rf.feature_importances_)]
print(f"  → Most important (forest): {top_rf}\n")

if top_tree == top_rf:
    print("  ✓  Both models agree on the most important feature.")
else:
    print("  ✗  The models disagree — worth investigating with more data.")
print()

# ── Bonus: vary n_estimators ──────────────────────────────────────────────────
print("── Bonus: accuracy vs n_estimators ───────────────────────────────────")
for n in [1, 5, 10, 50, 100]:
    rf_n = RandomForestClassifier(n_estimators=n, random_state=0)
    rf_n.fit(X_train, y_train)
    acc_n = (rf_n.predict(X_test) == y_test).mean() * 100
    print(f"  n_estimators={n:<4}: {acc_n:.0f}% accuracy")
print()

print("── Verdict ──────────────────────────────────────────────────────────")
print("  On this clean, linearly-separable dataset, even 1 tree achieves")
print("  perfect accuracy — so n_estimators barely changes the result here.")
print("  On messier, real-world data the forest's averaging removes noise")
print("  that trips up any single tree. caps_ratio is the dominant feature")
print("  in both models: a ratio above ~0.2 cleanly predicts spam.")
