import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('Gaming_Academic_Performance.csv')

df.drop(columns=['student_id'], errors='ignore', inplace=True)

for col in df.select_dtypes(include='number').columns:
    df[col].fillna(df[col].median(), inplace=True)
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

df['performance'] = pd.cut(df['grades'], bins=[0,50,75,100], labels=['Low','Medium','High'])
df['performance'] = LabelEncoder().fit_transform(df['performance'].astype(str))
df.drop(columns=['grades'], inplace=True)

X = df.drop(columns=['performance'])
y = df['performance']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f'Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')