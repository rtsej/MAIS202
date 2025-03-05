import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import ast

# Load the dataset (Modify the path as needed)
df = pd.read_csv(r"C:\Users\jcyr9\Downloads\Processed_Political_Bias_with_BERT (1).csv", encoding='utf-8')

print(repr(df.columns))
X = np.array(df["bert_embedding"]) 
# Check the column names and first few rows to verify the column exists
print(df.columns)  # Print column names
print(df.head())   # Print the first few rows

# Ensure 'bert_embedding' is a list of embeddings
# If 'bert_embedding' is a string representation of a list, convert it back
if 'bert_embedding' in df.columns:
    df["bert_embedding"] = df["bert_embedding"].apply(ast.literal_eval)
else:
    print("Column 'bert_embedding' not found. Please check the column name.")

# Drop rows with missing data in key columns
df = df.dropna(subset=["bert_embedding", "bias"])

# Assume 'bert_embedding' contains the text features and 'bias' is the label
X = np.array(df["bert_embedding"].tolist())  # Convert list of embeddings to NumPy array
y = df["bias"].values  # Target labels

# Split dataset: 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Train a Logistic Regression Model
model = LogisticRegression(max_iter=1000, solver="lbfgs")
model.fit(X_train, y_train)

# Evaluate on Validation Set
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)

# Evaluate on Test Set
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print Results
print("Validation Accuracy:", val_accuracy)
print("Test Accuracy:", test_accuracy)
print("\nClassification Report (Test Set):\n", classification_report(y_test, y_test_pred))
print("\nConfusion Matrix (Test Set):\n", confusion_matrix(y_test, y_test_pred))

# Check for Overfitting
train_accuracy = accuracy_score(y_train, model.predict(X_train))
print("\nTrain Accuracy:", train_accuracy)

# Overfitting Check
if train_accuracy > val_accuracy + 0.05:
    print("\n⚠️ Possible Overfitting Detected: Training accuracy is much higher than validation accuracy.")
elif val_accuracy > train_accuracy:
    print("\n⚠️ Possible Underfitting: Validation accuracy is higher than training accuracy.")
else:
    print("\n✅ Model is generalizing well.")
