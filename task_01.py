# 📦 Import Libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 🌿 Load Iris Dataset
iris = load_iris()
X = iris.data
y = iris.target
labels = iris.target_names

# 🔀 Split into Train/Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Train a Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Predictions and Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🌟 Accuracy: {accuracy:.2f}")
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred, target_names=labels))

# 🔍 Confusion Matrix Visualization
conf_mat = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(conf_mat, index=labels, columns=labels)

plt.figure(figsize=(6, 4))
sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='d')
plt.title('🔍 Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()