Features

Predict YouTube video views using ML

Feature engineering: title/description length, tags count, category, publish time

Regression models (Random Forest, Linear Regression, XGBoost)

Evaluate model using MSE, MAE, and R² Score

Beginner-friendly project

🛠 Technologies Used

Python 3

Pandas

Scikit-learn

XGBoost

NumPy

📁 Project Structure
youtube_video_prediction/
├── youtube_data.csv
├── model.py
├── README.md
└── predict.py

⚙️ Installation
pip install pandas scikit-learn xgboost numpy
How It Works
CSV Dataset → Preprocessing → Feature Engineering → ML Model Training → Prediction → Output

🧠 Model Explanation

Environment: Video metadata CSV

Sensor: Video attributes (title, description, tags, category, publish time)

Decision Maker: Machine Learning model

Actuator: Predicted views

💬 Working

Load dataset

Preprocess and extract features

Train regression model

Predict views for new video metadata

Evaluate model performance

📚 Use Cases

Student mini project

College AI/ML project

YouTube analytics prediction tool

Learning feature engineering and regression models

🚀 Future Enhancements

Include NLP features from title and description

Predict likes, comments, and engagement

Deploy as web application using Flask or Streamlit

Use real-time YouTube API data for live predictions

Incorporate deep learning models for higher accuracy

👩‍💻 Author

Vaishnavi Dhakane