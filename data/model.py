# model.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv("data/student_scores.csv")
X = data[['Hours']]
y = data['Scores']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

plt.scatter(X, y, label='Actual')
plt.plot(X, model.predict(X), label='Predicted')
plt.title("Hours vs Scores")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.legend()
plt.show()
