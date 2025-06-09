import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import accuracy_score

# ----------------------------
# Step 1: Load the Data
# ----------------------------
train = pd.read_csv(r"C:\Users\Rishabh Mishra\Documents\Kaggle\Fertilizer Prediction\playground-series-s5e6\train.csv")
test = pd.read_csv(r"C:\Users\Rishabh Mishra\Documents\Kaggle\Fertilizer Prediction\playground-series-s5e6\test.csv")
sample_submission = pd.read_csv(r"C:\Users\Rishabh Mishra\Documents\Kaggle\Fertilizer Prediction\playground-series-s5e6\sample_submission.csv")

print(train.head())
print(train.info())
print(train.describe())
print(train.columns)

# ----------------------------
# Step 2: Define Features and Target
# ----------------------------
X = train.drop(columns=['id', 'Fertilizer Name'])
y = train['Fertilizer Name']
X_test = test.drop(columns=['id'])

# ----------------------------
# Step 3: Handle Missing Values
# ----------------------------
X.fillna(-1, inplace=True)
X_test.fillna(-1, inplace=True)

# ----------------------------
# Step 4: Encode Categorical Features
# ----------------------------
categorical_cols = ['Soil Type', 'Crop Type']
le_dict = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])
    le_dict[col] = le

# ----------------------------
# Step 5: Split Train/Validation Data
# ----------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# Step 6: Train LightGBM Model
# ----------------------------
model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# ----------------------------
# Step 7: MAP@3 Evaluation Function
# ----------------------------
def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        if a in p[:k]:
            return 1 / (p.index(a) + 1)
        return 0
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

# Predict probabilities
probs = model.predict_proba(X_val)

# Get top 3 predictions
top_3 = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
class_names = model.classes_
top_3_labels = [[class_names[i] for i in row] for row in top_3]

# Evaluate MAP@3
score = mapk(y_val.values, top_3_labels, k=3)
print(f"Validation MAP@3 Score: {score:.4f}")

# ----------------------------
# Step 8: Feature Importance Plot
# ----------------------------
lgb.plot_importance(model, max_num_features=10)
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.show()

# ----------------------------
# Step 9: Predict on Test Data
# ----------------------------
test_probs = model.predict_proba(X_test)
top_3_test = np.argsort(test_probs, axis=1)[:, -3:][:, ::-1]
top_3_test_labels = [[class_names[i] for i in row] for row in top_3_test]

# Join top 3 predictions as space-separated strings
submission = sample_submission.copy()
submission['Fertilizer Name'] = [' '.join(preds) for preds in top_3_test_labels]

submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")
