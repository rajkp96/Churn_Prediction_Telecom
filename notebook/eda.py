import pandas as pd

df = pd.read_csv("C:/Users/Raj Kumar Patra/OneDrive/Documents/VS_code and Antigravity/churn_prediction_Telco/data/Telco_customer_churn_1.csv")
print(df.head())

df.shape
df.info()
print(df.describe(include='all'))

#missing value
print("missing values of all the column")
print(df.isnull().sum())

print("descriptive statistics")
print(df.describe())

#target variable distribution
print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize=True) * 100)

#graphical presentation
import matplotlib.pyplot as plt

df['Churn'].value_counts().plot(kind='bar')

plt.xlabel("Churn")
plt.ylabel("Count")
plt.title("Target Value Distribution")

plt.show()

categorical_cols=df.select_dtypes(include=['object']).columns.tolist()
print("\n Unique values in categorical column")
for col in categorical_cols:
    print(f"{col}:{df[col].unique()}")