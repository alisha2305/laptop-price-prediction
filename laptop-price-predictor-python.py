# Import necessary libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset
df = pd.read_csv(r"C:\Users\ALISHA\Downloads\laptop-price-predictor-regression-project-main\laptop-price-predictor-regression-project-main\laptop_data.csv")

# Drop unnecessary column
df.drop(columns=["Unnamed: 0"], inplace=True)

# Convert Ram and Weight to numeric
df["Ram"] = df["Ram"].str.replace("GB", "", regex=True).astype("int32")
df["Weight"] = df["Weight"].str.replace("kg", "", regex=True).astype("float32")

# Feature Engineering
df["Touchscreen"] = df["ScreenResolution"].apply(lambda x: 1 if "Touchscreen" in x else 0)
df["Ips"] = df["ScreenResolution"].apply(lambda x: 1 if "IPS" in x else 0)

# Extract resolution
res = df["ScreenResolution"].str.split("x", n=1, expand=True)
df["X_res"] = res[0].str.replace(",", "").str.findall(r"(\d+)").apply(lambda x: x[0]).astype("int")
df["Y_res"] = res[1].astype("int")

# Compute PPI
df["ppi"] = (((df["X_res"]**2) + (df["Y_res"]**2))**0.5 / df["Inches"]).astype("float")
df.drop(columns=["ScreenResolution", "Inches", "X_res", "Y_res"], inplace=True)

# Process CPU
df["Cpu Name"] = df["Cpu"].apply(lambda x: " ".join(x.split()[:3]))
def categorize_cpu(cpu):
    if cpu in ["Intel Core i3", "Intel Core i5", "Intel Core i7"]:
        return cpu
    elif "Intel" in cpu:
        return "Other Intel"
    else:
        return "AMD Processor"
df["Cpu brand"] = df["Cpu Name"].apply(categorize_cpu)
df.drop(columns=["Cpu", "Cpu Name"], inplace=True)

# Process Memory
df["Memory"] = df["Memory"].astype(str).replace(r"\.0", "", regex=True)
df["Memory"] = df["Memory"].str.replace("GB", "", regex=True).str.replace("TB", "000", regex=True)

# Split storage types
mem_split = df["Memory"].str.split("+", expand=True)

# Extract numeric HDD & SSD values
df["HDD"] = mem_split[0].apply(lambda x: int("".join(filter(str.isdigit, str(x)))) if "HDD" in str(x) else 0)
df["SSD"] = mem_split[0].apply(lambda x: int("".join(filter(str.isdigit, str(x)))) if "SSD" in str(x) else 0)

# Drop original Memory column
df.drop(columns=["Memory"], inplace=True)

# Process GPU
df["Gpu brand"] = df["Gpu"].apply(lambda x: x.split()[0])
df = df[df["Gpu brand"] != "ARM"]  # Remove ARM processors
df.drop(columns=["Gpu"], inplace=True)

# Process Operating System
def categorize_os(os):
    if os in ["Windows 10", "Windows 7", "Windows 10 S"]:
        return "Windows"
    elif os in ["macOS", "Mac OS X"]:
        return "Mac"
    else:
        return "Other"
df["os"] = df["OpSys"].apply(categorize_os)
df.drop(columns=["OpSys"], inplace=True)

# Prepare features and target
X = df.drop(columns=["Price"])
y = np.log(df["Price"])  # Log transformation

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

# Encoding categorical features
preprocessor = ColumnTransformer(transformers=[
    ("encoder", OneHotEncoder(sparse_output=False, drop="first"), [0, 1, 7, 10, 11])  # Categorical features
], remainder="passthrough")

# Linear Regression Model
lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)
r2_lr = r2_score(y_test, y_pred_lr)

# Decision Tree Model
dt_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeRegressor(max_depth=10, random_state=2))
])
dt_pipeline.fit(X_train, y_train)
y_pred_dt = dt_pipeline.predict(X_test)
r2_dt = r2_score(y_test, y_pred_dt)

# Compare accuracies
if r2_lr > r2_dt:
    best_model = lr_pipeline
    best_model_name = "Linear Regression"
    file_name = "best_model.pkl"
else:
    best_model = dt_pipeline
    best_model_name = "Decision Tree"
    file_name = "best_model.pkl"

# Save the best model
with open(file_name, "wb") as file:
    pickle.dump(best_model, file)

print(f"The best model is {best_model_name} with R² score: {max(r2_lr, r2_dt)}")
print(f"Model saved as {file_name}")
