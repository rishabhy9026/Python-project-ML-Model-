#libraries used in project.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#data set used in project.
df = pd.read_csv("C:/Users/ry376/Downloads/Python Dataset.csv")
df
df.head()
df.tail()
print("Shape:", df.shape)
print(df.columns)
df.info()
df.isnull().sum()
df['last_update'] = pd.to_datetime(df['last_update'], errors='coerce')
cols = ['pollutant_min', 'pollutant_max', 'pollutant_avg']
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df[cols] = df[cols].fillna(df[cols].mean())
df = df.drop_duplicates()
df['city'] = df['city'].str.lower().str.strip()
df['state'] = df['state'].str.lower().str.strip()
df.head()
df['year'] = df['last_update'].dt.year
df['month'] = df['last_update'].dt.month
df.describe()
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
for col in ['pollutant_min', 'pollutant_max', 'pollutant_avg']:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()   
for col in ['pollutant_min', 'pollutant_max', 'pollutant_avg']:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Outliers in {col}")
    plt.show()
plt.scatter(df['pollutant_min'], df['pollutant_avg'])
plt.xlabel("Pollutant Min")
plt.ylabel("Pollutant Avg")
plt.title("Min vs Avg Pollution")
plt.show()

plt.scatter(df['pollutant_max'], df['pollutant_avg'])
plt.xlabel("Pollutant Max")
plt.ylabel("Pollutant Avg")
plt.title("Max vs Avg Pollution")
plt.show()
sns.regplot(x='pollutant_min', y='pollutant_avg', data=df)
plt.title("Regression: Min vs Avg")
plt.show()
sns.pairplot(df[['pollutant_min', 'pollutant_max', 'pollutant_avg']])
plt.show()
top_states = df.groupby('state')['pollutant_avg'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=top_states.values, y=top_states.index)
plt.title("Top Polluted States")
plt.xlabel("Average Pollution")
plt.ylabel("State")
plt.show()
df.groupby('city')['pollutant_avg'].mean().sort_values(ascending=False).head(10).plot(kind='bar')
plt.title("Top Polluted Cities")
plt.show()
X = df[['pollutant_min', 'pollutant_max']]
y = df['pollutant_avg']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
baseline_pred = [y_train.mean()] * len(y_test)
baseline_score = r2_score(y_test, baseline_pred)

print("Baseline Accuracy (Before Training):", baseline_score)
model = LinearRegression()
model.fit(X_train, y_train)

print("Model Trained Successfully")
y_pred = model.predict(X_test)
model_score = r2_score(y_test, y_pred)

print("Model Accuracy (After Training):", model_score)
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()
sample = [[30, 100]]
print("Predicted Pollution Avg:", model.predict(sample))
