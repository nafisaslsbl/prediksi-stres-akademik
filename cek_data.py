import pandas as pd
df = pd.read_csv("data/university_student_stress_dataset.csv")
print(df.columns.tolist())
print(df.head())
print(df.dtypes)