# 🧪 Fertilizer Prediction -

This project contains a machine learning pipeline to solve the [Kaggle Playground Series - Season 5, Episode 6](https://www.kaggle.com/competitions/playground-series-s5e6) competition. The goal is to predict the most suitable **fertilizer** based on soil and environmental features.

---

##  Problem Statement

Given a dataset with features such as:

- `Temparature`
- `Humidity`
- `Moisture`
- `Soil Type` (categorical)
- `Crop Type` (categorical)
- `Nitrogen`
- `Phosphorous`
- `Potassium`

The task is to predict the correct **Fertilizer Name** from a fixed list of classes. Each prediction must include a **ranked list of top 3 fertilizers**, evaluated using **Mean Average Precision at 3 (MAP@3)**.

---

##  Solution Overview

###  Preprocessing
- Missing values handled with `fillna(-1)`
- Categorical features (`Soil Type`, `Crop Type`) encoded using `LabelEncoder`

###  Model
- Used `LightGBMClassifier` for training
- Evaluation metric: **MAP@3**
- Also tracked accuracy as a baseline metric

###  Submission Format
- Outputs a CSV file with:
  - `id`
  - `Fertilizer Name`: top 3 predicted fertilizers (space-separated)

---

## Project Structure

fertilizer-prediction/
│
├── main.py # Full pipeline: load, preprocess, train, evaluate, submit
├── submission.csv # Final Kaggle submission file
├── README.md # This file
└── data/
├── train.csv
├── test.csv
└── sample_submission.csv

## 📊 MAP@3 Metric Explained

MAP@3 rewards you for getting the correct fertilizer within the **top 3 predictions**:

- Score = `1` if correct fertilizer is first
- Score = `1/2` if second, `1/3` if third
- Mean is taken over all rows

### MAP@3 Implementation:
```python
def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        if a in p[:k]:
            return 1 / (p.index(a) + 1)
        return 0
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

**### Feature Importance**
Feature importance was extracted using LightGBM’s built-in function to visualize key contributors to the model’s predictions.

Example Plot:
import matplotlib.pyplot as plt
lgb.plot_importance(model, max_num_features=10)
plt.show()

**How to Run Locally**

1. Clone the repo:

git clone https://github.com/yourusername/fertilizer-prediction.git
cd fertilizer-prediction
2. Install dependencies:

pip install -r requirements.txt
If you don’t have a requirements.txt, install manually:


pip install pandas numpy scikit-learn lightgbm matplotlib
3. Run the script:

python main.py
4. Submit the output:
Upload submission.csv to Kaggle S5E6

📦 Dependencies
Python 3.8+
pandas
numpy
scikit-learn
lightgbm
matplotlib


📚 References
Kaggle Competition Page

LightGBM Docs

MAP@K Metric

🧑‍💻 Author
Rishabh Mishra
Email: Mishra.rishabh11@gmail.com

📜 License
This project is open source under the MIT License.
