# These are the essential Python libraries for data handling, preprocessing, and running machine learning models.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Load the dataset
df = pd.read_csv('breast-cancer.csv')

# Drop the 'id' column if present (not useful for prediction)
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# Encode the target column 'diagnosis' (M = 1, B = 0)
label_encoder = LabelEncoder()
df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])

# Separate features and target, y= diag and x daig column dropped
X = df.drop(columns=['diagnosis'])  # Features
y = df['diagnosis']                 # Target

# Normalize features to similar scale (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
# ------------------------------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
print(f"Decision Tree Accuracy: {dt_acc:.2f}")

# Naive Bayes Classifier
# ------------------------------
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
nb_acc = accuracy_score(y_test, nb_pred)
print(f"Naive Bayes Accuracy: {nb_acc:.2f}")

# Neural Network (MLP Classifier)
# ------------------------------
nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
nn.fit(X_train, y_train)
nn_pred = nn.predict(X_test)
nn_acc = accuracy_score(y_test, nn_pred)
print(f"Neural Network Accuracy: {nn_acc:.2f}")
