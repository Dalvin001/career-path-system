# predict.py
# ====================
import sys
import pickle
import numpy as np

# --------------------------
# Load Models
with open('./model1.pkl', 'rb') as f:
    clf1 = pickle.load(f)

with open('./model2.pkl', 'rb') as f:
    clf2 = pickle.load(f)

with open('./model3.pkl', 'rb') as f:
    clf3 = pickle.load(f)

with open('./model4.pkl', 'rb') as f:
    clf4 = pickle.load(f)

# Load Label Encoder to decode predictions
with open('./label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# --------------------------
# Collect user input from command line arguments
# Example usage: python predict.py 7 8 2 5 1 0 1 0 1 2 1 0 1 0 1 2 0 1 1 0 2
userdata = [sys.argv[1:]]

# Convert all input strings to float (or int) as required
userdata = np.array(userdata, dtype=float)

# --------------------------
# Predictions

# Decision Tree
pred1_encoded = clf1.predict(userdata)
pred1 = le.inverse_transform(pred1_encoded)
probs1 = np.max(clf1.predict_proba(userdata))

# SVM
pred2_encoded = clf2.predict(userdata)
pred2 = le.inverse_transform(pred2_encoded)
probs2 = np.max(clf2.predict_proba(userdata))

# Random Forest
pred3_encoded = clf3.predict(userdata)
pred3 = le.inverse_transform(pred3_encoded)
probs3 = np.max(clf3.predict_proba(userdata))

# XGBoost
pred4_encoded = clf4.predict(userdata)
pred4 = le.inverse_transform(pred4_encoded)
probs4 = np.max(clf4.predict_proba(userdata))

# --------------------------
# Print Results
print("Decision Tree Prediction:", pred1[0], "with probability:", probs1)
print("SVM Prediction:", pred2[0], "with probability:", probs2)
print("Random Forest Prediction:", pred3[0], "with probability:", probs3)
print("XGBoost Prediction:", pred4[0], "with probability:", probs4)
