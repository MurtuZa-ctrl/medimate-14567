"""
MediMate - missed-dose risk model (Adherence Agent)

Trains a small decision tree that estimates the probability that a patient
will miss the NEXT scheduled dose. The model is exported to model.json and
embedded in index.html so the app runs in the browser with no server.

IMPORTANT: the training data is SYNTHETIC. It is generated from assumptions
(older age, more daily doses, living alone, recent misses, late-night doses
and weekends raise the chance of a miss). This is a prototype. A real
deployment must be trained and validated on consented, real adherence logs.
"""
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

rng = np.random.default_rng(42)
N = 6000

age = rng.integers(40, 91, N)
doses_per_day = rng.integers(1, 6, N)
lives_alone = rng.integers(0, 2, N)
missed_last_7d = np.clip(rng.poisson(1.2 + 0.03 * (age - 40) + 0.3 * lives_alone), 0, 14)
hour = rng.choice([7, 8, 9, 13, 14, 18, 20, 21, 22], N)
weekend = rng.integers(0, 2, N)

# Assumed risk pattern used to generate synthetic labels
z = (-3.0
     + 0.025 * (age - 60)
     + 0.30 * (doses_per_day - 2)
     + 0.55 * lives_alone
     + 0.28 * missed_last_7d
     + 0.55 * (hour >= 21)
     + 0.25 * (hour <= 7)
     + 0.35 * weekend)
p_miss = 1 / (1 + np.exp(-z))
y = (rng.random(N) < p_miss).astype(int)

X = np.column_stack([age, doses_per_day, lives_alone, missed_last_7d, hour, weekend])
features = ["age", "doses_per_day", "lives_alone", "missed_last_7d", "hour", "weekend"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)
clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=60, random_state=7)
clf.fit(X_tr, y_tr)

acc = accuracy_score(y_te, clf.predict(X_te))
auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
base = max(y_te.mean(), 1 - y_te.mean())
print(f"test accuracy {acc:.3f} | AUC {auc:.3f} | majority-class baseline {base:.3f} | miss rate {y.mean():.3f}")


def export(node=0):
    t = clf.tree_
    if t.children_left[node] == -1:
        v = t.value[node][0]
        return {"p": round(float(v[1] / v.sum()), 4)}
    return {
        "f": int(t.feature[node]),
        "t": round(float(t.threshold[node]), 3),
        "l": export(t.children_left[node]),
        "r": export(t.children_right[node]),
    }


model = {
    "features": features,
    "tree": export(),
    "meta": {
        "type": "DecisionTreeClassifier(max_depth=4, min_samples_leaf=60)",
        "train_rows": int(len(y_tr)),
        "test_rows": int(len(y_te)),
        "test_accuracy": round(float(acc), 3),
        "test_auc": round(float(auc), 3),
        "baseline_accuracy": round(float(base), 3),
        "data": "synthetic",
    },
}
with open("model.json", "w") as f:
    json.dump(model, f)
print("saved model.json, leaves:", int(clf.get_n_leaves()), "depth:", int(clf.get_depth()))
