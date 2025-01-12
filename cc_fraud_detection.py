import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import tkinter as tk
from tkinter import Label, Entry, Button
import time

# Load dataset
start_time = time.time()
data = pd.read_csv("C:\\Users\\DELL\\OneDrive\\Desktop\\SEMESTER_IV(PROJECT)\\creditcard.csv")
print("Dataset loaded successfully.", flush=True)

# Data overview
print("Number of Rows", data.shape[0], flush=True)
print("Number of Columns", data.shape[1], flush=True)
data.info()

# Check for null values
if data.isnull().sum().any():
    print("Null values found in the dataset.", flush=True)
else:
    print("No null values in the dataset.", flush=True)

# Standardize 'Amount' column
sc = StandardScaler()
data['Amount'] = sc.fit_transform(pd.DataFrame(data['Amount']))

# Drop 'Time' column
data = data.drop(['Time'], axis=1)
print("Dropped 'Time' column.", flush=True)

# Remove duplicates
if data.duplicated().any():
    data = data.drop_duplicates()
    print("Duplicates removed.", flush=True)

# Plot class distribution
sns.countplot(x='Class', data=data)
plt.title("Class Distribution")
plt.show(block=False)

# Handle class imbalance using SMOTE
x = data.drop('Class', axis=1)
y = data['Class']
smote = SMOTE()
x_res, y_res = smote.fit_resample(x, y)
print("Applied SMOTE for class balancing.", flush=True)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.20, random_state=42)
print("Dataset split into training and testing sets.", flush=True)

# Model training and evaluation
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

results = []
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append({'Model': name, 'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1 Score': f1})
    print(f"{name} trained. Accuracy: {acc:.2f}", flush=True)

# Display results
results_df = pd.DataFrame(results)
print("\nModel Performance:\n", results_df, flush=True)

# Plot results
sns.barplot(x='Model', y='Accuracy', data=results_df)
plt.title("Model Accuracy Comparison")
plt.show(block=False)

# Save the best model (Random Forest in this case)
best_model = models['Random Forest']
joblib.dump(best_model, "Credit_card_model.pkl")
print("Best model saved as 'Credit_card_model.pkl'.", flush=True)

# Load and test saved model
model = joblib.load("Credit_card_model.pkl")
test_prediction = model.predict([x_test.iloc[0]])
print("Test prediction on a sample: ", "Fraudulent" if test_prediction[0] == 1 else "Normal", flush=True)

# # GUI for fraud detection
# def predict_transaction():
#     try:
#         values = [float(entries[i].get()) for i in range(29)]  # Collect values from Entry widgets
#         prediction = model.predict([values])
#         result_label.config(text="Fraudulent Transaction" if prediction[0] == 1 else "Normal Transaction")
#     except Exception as e:
#         result_label.config(text=f"Error: {e}")

# # Create Tkinter GUI
# root = tk.Tk()
# root.title("Credit Card Fraud Detection System")

# Label(root, text="Credit Card Fraud Detection System", bg="black", fg="white", width=40).grid(row=0, columnspan=2, pady=10)

# entries = []
# for i in range(1, 30):
#     Label(root, text=f"Enter value of V{i}").grid(row=i, column=0, padx=10, pady=5, sticky="e")
#     entry = Entry(root)
#     entry.grid(row=i, column=1, padx=10, pady=5)
#     entries.append(entry)

# Button(root, text="Predict", command=predict_transaction).grid(row=30, column=0, columnspan=2, pady=10)
# result_label = Label(root, text="", font=("Helvetica", 12))
# result_label.grid(row=31, column=0, columnspan=2, pady=10)

# print("Starting GUI...", flush=True)
# root.mainloop()

# GUI for fraud detection with scroll functionality
def predict_transaction():
    try:
        values = [float(entries[i].get()) for i in range(29)]  # Collect values from Entry widgets
        prediction = model.predict([values])
        result_label.config(text="Fraudulent Transaction" if prediction[0] == 1 else "Normal Transaction")
    except Exception as e:
        result_label.config(text=f"Error: {e}")

# Create Tkinter GUI with scrollable frame
root = tk.Tk()
root.title("Credit Card Fraud Detection System")
root.geometry("400x600")  # Set a default size for the window

# Create a canvas and scrollbar
canvas = tk.Canvas(root)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

# Configure canvas
scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

# Pack canvas and scrollbar
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Add widgets to the scrollable frame
Label(scrollable_frame, text="Credit Card Fraud Detection System", bg="black", fg="white", width=40).grid(row=0, columnspan=2, pady=10)

entries = []
for i in range(1, 30):
    Label(scrollable_frame, text=f"Enter value of V{i}").grid(row=i, column=0, padx=10, pady=5, sticky="e")
    entry = Entry(scrollable_frame)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries.append(entry)

Button(scrollable_frame, text="Predict", command=predict_transaction).grid(row=30, column=0, columnspan=2, pady=10)
result_label = Label(scrollable_frame, text="", font=("Helvetica", 12))
result_label.grid(row=31, column=0, columnspan=2, pady=10)

print("Starting GUI...", flush=True)
root.mainloop()
