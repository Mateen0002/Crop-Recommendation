import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Step 1: Load the dataset
df = pd.read_csv('Cropdataset.csv')

# Step 2: Split features and target
X = df.drop('label', axis=1)
y = df['label']

# Step 3: Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Step 4: Save the model using pickle
pickle.dump(model, open('crop_model.pkl', 'wb'))

print("Model trained and saved as crop_model.pkl")
