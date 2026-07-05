
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data/retail_sales_dataset.csv')

print("Dataset Shape:", df.shape)
print(df.describe())

plt.figure(figsize=(6,4))
sns.barplot(x='Month', y='Sales', data=df)
plt.title('Monthly Sales')
plt.savefig('output/monthly_sales.png')
plt.close()

plt.figure(figsize=(6,4))
sns.scatterplot(x='Customers', y='Profit', data=df)
plt.title('Customers vs Profit')
plt.savefig('output/customers_profit.png')
plt.close()

X = df[['Sales','Customers']]
y = df['Profit']

model = LinearRegression()
model.fit(X,y)

predictions = model.predict(X)

plt.figure(figsize=(6,4))
plt.plot(df['Month'], y, marker='o', label='Actual Profit')
plt.plot(df['Month'], predictions, marker='s', label='Predicted Profit')
plt.legend()
plt.title('Profit Prediction')
plt.savefig('output/profit_prediction.png')
plt.close()

with open('output/project_report.txt','w') as f:
    f.write('Retail Sales Analysis Completed\n')
    f.write('Model: Linear Regression\n')

print('Project Completed Successfully')
