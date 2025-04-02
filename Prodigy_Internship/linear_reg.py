import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

#load the kaggle_data

train_data = pd.read_csv("//train.csv")
print("Kaggle house prices (train.csv) preview:")
print(train_data.head())
print(train_data.info())

#load Mumbai hosuing price data
Mumbai_data = pd.read_csv("//Mumbai House Prices.csv")
print("\nMumbai house prices preview:")
print(Mumbai_data.head())
print(Mumbai_data.info())

#CHECKING FOR MISSING VALUES
print("missing values in train_data:")
print(train_data.isnull().sum().sort_values(ascending=False).head(10))

#let fill the mising  values with  medain and mode for numeric data
numeric_cols = train_data.select_dtypes(include=['float64', 'int64']).columns
train_data[numeric_cols] = train_data[numeric_cols].fillna(train_data[numeric_cols].median())

#for train_data

train_data['HouseAge'] = train_data['YrSold'] - train_data['YearBuilt']
train_data['PricePerSqFt'] = train_data['SalePrice']/train_data['GrLivArea']

#For India
np.random.seed(42)
train_data['City'] = np.random.choice(['Mumbai','Delhi','Bangalore'],size = len(train_data))

#create a simulated RBI DAta.
rbi_data = pd.DataFrame({'City': ['Mumbai','Delhi','Bangalore'],
                          'HousingIndex': [400,350,360]
                          })

#merge the RBI data into train_data

train_data = pd.merge(train_data,rbi_data, on='City', how='left')
train_data['AdjustedPrice'] = train_data['SalePrice'] / train_data['HousingIndex']

#Preview the new feature
print("New feature added to train_data")
print(train_data[['HouseAge','PricePerSqFt','City','HousingIndex','AdjustedPrice']].head())

#Choose  your feature and target

features = train_data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HouseAge', 'PricePerSqFt', 'AdjustedPrice']]
target = train_data['SalePrice']


#split the dATASET INTO  TRAINING AND TESTING
X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.2,random_state=42)

#BULID THE LINEAR REG

model =LinearRegression()
model.fit(X_train,y_train)

#make the prediction and evaluate the model

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
print("Mean Squared Error",mse)

varOcg = mse
print("varOcg (MSE):",varOcg)

sns.set(style="darkgrid", palette="viridis")
plt.figure(figsize=(12, 9))

# Get city information for the test set rows
city_test = train_data.loc[X_test.index, 'City']

# Create scatter plot with hue based on city
scatter = sns.scatterplot(x=y_test, y=y_pred, hue=city_test, s=80, alpha=0.85, edgecolor='w', linewidth=0.7)

plt.xlabel("Actual Sale Price", fontsize=16, weight='bold', color='navy')
plt.ylabel("Predicted Sale Price", fontsize=16, weight='bold', color='navy')
plt.title("Actual vs Predicted House Prices with City Annotations", fontsize=20, weight='bold', color='darkred')

# Draw ideal fit line (diagonal)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label="Ideal Fit")

# Annotate the first 5 test points with their city names
for j, i in enumerate(X_test.index[:5]):
    plt.text(y_test.loc[i], y_pred[j], f"{train_data.loc[i, 'City']}",
             fontsize=12, color='black', weight='bold',
             bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='none'))

plt.legend(title="City", fontsize=12, title_fontsize=14)
plt.tight_layout()
plt.show()



