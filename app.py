# app.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import sys

# ---------------------------
# Step 1: Load Dataset
# ---------------------------
try:
    data = pd.read_csv("youtube_data.csv")
except FileNotFoundError:
    print("Error: 'youtube_data.csv' not found in the current folder.")
    sys.exit(1)

# Check required columns
required_columns = ['title_length', 'description_length', 'tags_count', 'category', 'publish_hour', 'views']
for col in required_columns:
    if col not in data.columns:
        print(f"Error: Missing required column '{col}' in CSV.")
        sys.exit(1)

# Drop rows with missing values
data.dropna(subset=required_columns, inplace=True)

# ---------------------------
# Step 2: Prepare Features & Target
# ---------------------------
X = data[['title_length', 'description_length', 'tags_count', 'category', 'publish_hour']]
y = data['views']

# ---------------------------
# Step 3: Split Dataset
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# Step 4: Train Model
# ---------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------------
# Step 5: Evaluate Model
# ---------------------------
y_pred = model.predict(X_test)
print("\n=== Model Evaluation Metrics ===")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# ---------------------------
# Step 6: Save Model
# ---------------------------
with open("youtube_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model trained and saved as 'youtube_model.pkl'.")

# ---------------------------
# Step 7: Predict New Video Views
# ---------------------------
def predict_views(title_length, description_length, tags_count, category, publish_hour):
    try:
        new_video = pd.DataFrame([{
            'title_length': int(title_length),
            'description_length': int(description_length),
            'tags_count': int(tags_count),
            'category': int(category),
            'publish_hour': int(publish_hour)
        }])
        prediction = model.predict(new_video)
        return int(prediction[0])
    except ValueError:
        print("Error: Invalid input. Please enter integers only.")
        return None

# ---------------------------
# Step 8: Interactive Prediction
# ---------------------------
if __name__ == "__main__":
    print("\n=== YouTube Video Views Prediction ===")
    try:
        title_len = int(input("Enter Title Length: "))
        desc_len = int(input("Enter Description Length: "))
        tags = int(input("Enter Number of Tags: "))
        category = int(input("Enter Category ID (integer): "))
        hour = int(input("Enter Publish Hour (0-23): "))

        predicted = predict_views(title_len, desc_len, tags, category, hour)
        if predicted is not None:
            print(f"\nPredicted Views for this video: {predicted}")
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except ValueError:
        print("Error: Please enter valid integer values.")
