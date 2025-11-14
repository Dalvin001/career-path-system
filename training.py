# training.py
# ====================
# Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree, svm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

print("Loading dataset...")
df = pd.read_csv("../data/mldata.csv")  # make sure path is correct

# --------------------------
# 1. Binary Encoding
binary_cols = ["self-learning capability?", "Extra-courses did",
               "Taken inputs from seniors or elders", "worked in teams ever?", "Introvert"]
for col in binary_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})

# 2. Number Encoding
num_cols = ["reading and writing skills", "memory capability score"]
num_map = {"poor": 0, "medium": 1, "excellent": 2}
for col in num_cols:
    df[col] = df[col].map(num_map)

# 3. Label Encoding for categorical columns
category_cols = ['certifications', 'workshops', 'Interested subjects',
                 'interested career area ', 'Type of company want to settle in?', 'Interested Type of Books']
for col in category_cols:
    df[col] = df[col].astype('category')
    df[col + "_code"] = df[col].cat.codes

# 4. Dummy variable encoding
df = pd.get_dummies(df, columns=["Management or Technical", "hard/smart worker"], prefix=["A", "B"])

# --------------------------
# Features and target
features = ['Logical quotient rating', 'coding skills rating', 'hackathons', 'public speaking points',
            'self-learning capability?', 'Extra-courses did', 'Taken inputs from seniors or elders',
            'worked in teams ever?', 'Introvert', 'reading and writing skills', 'memory capability score',
            'B_hard worker', 'B_smart worker', 'A_Management', 'A_Technical', 'Interested subjects_code',
            'Interested Type of Books_code', 'certifications_code', 'workshops_code',
            'Type of company want to settle in?_code', 'interested career area _code']

target = 'Suggested Job Role'

X = df[features]
y = df[target]

# Encode target labels (required for XGBoost)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save label encoder for future use
with open('../label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# --------------------------
# Model Training
print("Training Decision Tree...")
clf1 = tree.DecisionTreeClassifier()
clf1.fit(x_train, y_train)

print("Training SVM...")
clf2 = svm.SVC(probability=True)  # add probability=True to get predict_proba
clf2.fit(x_train, y_train)

print("Training Random Forest...")
clf3 = RandomForestClassifier(n_estimators=100)
clf3.fit(x_train, y_train)

print("Training XGBoost...")
clf4 = XGBClassifier(random_state=42, learning_rate=0.02, n_estimators=300)
clf4.fit(x_train, y_train)

# --------------------------
# Save models
model_files = ['../model1.pkl', '../model2.pkl', '../model3.pkl', '../model4.pkl']
models = [clf1, clf2, clf3, clf4]

for model, file in zip(models, model_files):
    with open(file, 'wb') as f:
        pickle.dump(model, f)

print("All models trained and saved successfully!")
