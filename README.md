# PRODIGY_ML_01
House Price Prediction Using Linear Regression
Overview
This project implements a linear regression model to predict house prices using features such as square footage, the number of bedrooms, and bathrooms. Additional feature engineering (e.g., house age, price per square foot) and market index adjustments help enhance prediction accuracy.

Datasets
Kaggle House Prices Dataset: House Prices Advanced Regression Techniques

Mumbai House Prices Dataset: Provides complementary data specific to Mumbai.

Project Structure
bash
Copy
Edit
HousePricePrediction/
├── data/
│   ├── train.csv
│   └── Mumbai House Prices.csv
├── src/
│   └── linear_regression.py   # Contains preprocessing, feature engineering, model training, and evaluation
├── README.md
└── requirements.txt
Requirements
Python 3.7+

pandas

numpy

scikit-learn

matplotlib

seaborn

Install the dependencies with:

bash
Copy
Edit
pip install -r requirements.txt
Usage
Data Preprocessing & Feature Engineering:
The script reads the datasets, fills missing values, and generates new features like HouseAge, PricePerSqFt, and an AdjustedPrice based on market indices.

Model Training:
A Linear Regression model is trained using features such as GrLivArea, BedroomAbvGr, and FullBath, among others.

Evaluation:
The model’s performance is evaluated using Mean Squared Error (MSE). A scatter plot is generated to compare actual vs. predicted prices with city annotations.

Run the project with:

bash
Copy
Edit
python src/linear_regression.py
Results
The model provides predictions with a quantifiable error metric (MSE) and visualizes the accuracy of predictions against actual prices.

Future Work
Explore regularized regression techniques (Ridge, Lasso) for further performance improvements.

Incorporate additional features and advanced feature engineering.
