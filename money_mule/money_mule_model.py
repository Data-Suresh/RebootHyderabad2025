
import pandas as pd
import numpy as np
import joblib
import json
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers

# Load the dataset
df = pd.read_csv("synthetic_money_mule.csv")

# Simulate labels for demonstration
np.random.seed(42)
df['label'] = np.random.randint(0, 2, size=len(df))

# Normalize all features (excluding label and CCR_grade)
features = df.drop(columns=['label'], errors='ignore').copy()

# Save the original feature column names used for scaling and inference
feature_columns = features.columns.tolist()
with open("feature_columns.json", "w") as f:
    json.dump(feature_columns, f)

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(features)

# Autoencoder model
input_dim = X_scaled.shape[1]
encoding_dim = 6
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu',
                activity_regularizer=regularizers.l1(1e-5))(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train autoencoder
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, shuffle=True, verbose=0)

# Reconstruction error
X_pred = autoencoder.predict(X_scaled)
df['reconstruction_error'] = np.mean(np.square(X_scaled - X_pred), axis=1)

# Intent Features Score
df['intent_score'] = df['reversal_flag'] + df['round_number_flag']

# IAI = α * Reconstruction Error + β * Intent Score
alpha, beta = 0.6, 0.4
df['IAI'] = alpha * df['reconstruction_error'] + beta * df['intent_score']
reconstruction_scaler = MinMaxScaler()

# CCR = 1 - Normalized Avg. Reconstruction Error
df['reconstruction_error_scaled'] = reconstruction_scaler.fit_transform(df[['reconstruction_error']])
df['CCR'] = 1 - df['reconstruction_error_scaled']

# CCR Grade
def grade_ccr(ccr):
    if ccr >= 0.75:
        return 'A'
    elif ccr >= 0.5:
        return 'B'
    elif ccr >= 0.25:
        return 'C'
    else:
        return 'D'

df['CCR_grade'] = df['CCR'].apply(grade_ccr)

# Prepare final features for classifier
X = df.drop(columns=['label', 'CCR_grade','reconstruction_error_scaled'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Features used to train RF model:", X_train.columns.tolist())


# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
with open("rf_feature_columns.json", "w") as f:
    json.dump(X_train.columns.tolist(), f)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Save reports
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
df.to_csv("money_mule_with_IAI_CCR.csv", index=False)
report_df.to_csv("classification_report.csv")

# Save models
autoencoder.save("autoencoder_model.keras")
joblib.dump(clf, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
# Save this separate scaler
joblib.dump(reconstruction_scaler, "reconstruction_scaler.pkl")

print("Training complete. Models and column info saved.")
