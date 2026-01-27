#!/usr/bin/env python
"""
Train Intent Gate classifier
"""

import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.data.fashioniq_loader import FashionIQLoader
from src.models.clip_encoder import CLIPEncoder
from src.models.intent_gate import IntentGate

print("=" * 70)
print("TRAINING INTENT GATE")
print("=" * 70)

# Load data and embeddings
print("\n[1/5] Loading data and embeddings...")
loader = FashionIQLoader(max_samples=5000)
texts, images = loader.load()

encoder = CLIPEncoder(device='cpu')
text_emb = encoder.encode_text(texts)
image_emb = encoder.encode_images(images)

# Extract gate features
print("\n[2/5] Extracting gate features...")
gate = IntentGate(num_queries=len(texts))
gate_features = gate.extract_features(texts, text_emb, image_emb)
print(f"Features shape: {gate_features.shape}")

# Load oracle labels
print("\n[3/5] Loading oracle labels...")
with open('data/processed/oracle_labels.json', 'r') as f:
    oracle_labels = json.load(f)

# Convert oracle alphas to class labels
alpha_to_class = {0.0: 0, 0.25: 1, 0.5: 2, 0.75: 3, 1.0: 4}
labels = np.array([alpha_to_class[oracle_labels[str(i)]['alpha']] for i in range(len(texts))])

print(f"Label distribution:")
for class_id, alpha in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
    count = sum(labels == class_id)
    print(f"  α={alpha}: {count} queries")

# Split data
print("\n[4/5] Splitting data (80/20 train/test)...")
X_train, X_test, y_train, y_test = train_test_split(
    gate_features, labels, test_size=0.2, random_state=42
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train classifier
print("\nTraining RandomForest classifier...")
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
print("\n[5/5] Evaluating...")
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
# Only show classes that appear in the test set
unique_labels = np.unique(np.concatenate([y_test, y_pred]))
class_names_subset = [['α=0.0', 'α=0.25', 'α=0.5', 'α=0.75', 'α=1.0'][i] for i in unique_labels]
print(classification_report(y_test, y_pred, labels=unique_labels, target_names=class_names_subset))

# Feature importance
print("\nFeature Importance:")
feature_names = ['caption_len', 'has_color', 'has_attr', 'ambiguity', 'sim_gap', 'specificity', 'dummy']
for name, importance in zip(feature_names, clf.feature_importances_):
    print(f"  {name:<15}: {importance:.4f}")

# Save model
import pickle
with open('models/intent_gate_trained.pkl', 'wb') as f:
    pickle.dump((clf, scaler), f)

print("\n" + "=" * 70)
print("✓ Gate trained and saved to: models/intent_gate_trained.pkl")
print(f"✓ Accuracy: {accuracy*100:.2f}%")
print("=" * 70)
print("\nAcceptance criteria:")
print(f"  ✓ Accuracy ≥ 85%? {'YES' if accuracy >= 0.85 else 'NO'}")
print("\nNext: Validate gate on test set and compare with oracle")
