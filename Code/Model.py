# FNN Model Code for Complete Recommendation system
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Set random seeds for reproducibility
import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(42)

# ===============================
# 1. Define Thresholds
# ===============================
MOISTURE_THRESHOLDS = {'low': 0, 'medium': 25, 'high': 28}
OIL_THRESHOLDS = {
    'fair': {'low': 23, 'medium': 26.5, 'high': 30},
    'light': {'low': 22, 'medium': 26, 'high': 30},
    'medium': {'low': 21, 'medium': 24.5, 'high': 28},
    'tan_brown': {'low': 20, 'medium': 23, 'high': 26},
    'deep_brown_black': {'low': 19, 'medium': 22, 'high': 25},
    'olive': {'low': 22, 'medium': 25.5, 'high': 29}
}

def get_level(value, thresholds, skin_tone=None):
    """Determine level (low/medium/high) based on value and thresholds"""
    if skin_tone and isinstance(thresholds, dict) and skin_tone in thresholds:
        thresh = thresholds[skin_tone]
    else:
        thresh = thresholds
        
    if value < thresh['medium']:
        return 'low'
    elif value < thresh['high']:
        return 'medium'
    else:
        return 'high'

# ===============================
# 2. Load and Preprocess Dataset
# ===============================
try:
    # Load the dataset
    data = pd.read_csv(r"C:\MiDerma\tone_adjusted_oil_dataset.csv")
    
    # Clean skin tone names
    data['SkinTone'] = data['SkinTone'].replace({'deep_brow': 'deep_brown_black'})
    
    # Encode skin tones
    le_skin = LabelEncoder()
    data['SkinTone_encoded'] = le_skin.fit_transform(data['SkinTone'])
    
    # Create recommendation classes
    data['RecClass'] = (
        data['SkinTone'] + '_' +
        data['Moisture'].apply(lambda x: get_level(x, MOISTURE_THRESHOLDS)) + '_' +
        data.apply(lambda row: get_level(row['Oil'], OIL_THRESHOLDS, row['SkinTone']), axis=1)
    )
    
    # Encode labels
    le_label = LabelEncoder()
    labels_encoded = le_label.fit_transform(data['RecClass'])
    num_classes = len(le_label.classes_)
    y_cat = to_categorical(labels_encoded, num_classes)
    
    # Scale features
    scaler = MinMaxScaler()
    features = scaler.fit_transform(data[['SkinTone_encoded', 'Moisture', 'Oil']].values)
    
except Exception as e:
    print(f"Error in data preprocessing: {str(e)}")
    raise

# ===============================
# 3. Visualize Class Distribution
# ===============================
plt.figure(figsize=(15, 5))
pd.Series(labels_encoded).value_counts().plot(kind='bar')
plt.title("Class Distribution")
plt.xlabel("Encoded Classes")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ===============================
# 4. Split Dataset
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    features, y_cat, 
    test_size=0.2, 
    random_state=42, 
    stratify=labels_encoded
)

# Calculate class weights for imbalanced classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels_encoded),
    y=labels_encoded
)
class_weight_dict = dict(enumerate(class_weights))

# ===============================
# 5. Build Model
# ===============================
model = Sequential([
    Input(shape=(3,)),
    Dense(64, activation='relu'), 
    BatchNormalization(),
    Dropout(0.2), 
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dense(num_classes, activation='softmax')
])


model.compile(
    optimizer=Adam(learning_rate=0.0005), 
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,  
    restore_best_weights=True,
    min_delta=0.001  # Minimum change to qualify as an improvement
)


reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.00001
)

# ===============================
# 6. Train Model
# ===============================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,  
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr],
    shuffle=True
)

# ===============================
# 7. Plot Training History
# ===============================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy plot
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

# Loss plot
ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

plt.tight_layout()
plt.show()

# ===============================
# 8. Evaluate Model
# ===============================
# Get predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Print classification report
print("\nClassification Report:")
print(classification_report(
    y_test_classes, 
    y_pred_classes,
    target_names=le_label.classes_
))

# Plot confusion matrix
plt.figure(figsize=(15, 15))
cm = confusion_matrix(y_test_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_label.classes_,
            yticklabels=le_label.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()


# ===============================
# 10. Save Model
# ===============================
# Save the trained model
model.save('skincare_recommendation_model.h5')
