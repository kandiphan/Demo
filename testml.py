import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

# ===================== LOAD DATA =====================
df = pd.read_csv("filegopchoml_final_clean.csv")

features = ["ROA", "DE", "BV", "PB"]
target = "EPS"

df = df.dropna(subset=features + [target]).reset_index(drop=True)

print("Số mã hợp lệ:", len(df))

# ===================== X / y =====================
X = df[features]
y = df[target]

# ===================== SPLIT (CHỈ ĐỂ KIỂM TRA MODEL) =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ===================== MODEL =====================
model = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    ))
])

# ===================== TRAIN =====================
model.fit(X_train, y_train)

# ===================== CHECK MODEL (THAM KHẢO) =====================
y_pred = model.predict(X_test)

print("R2 (tham khảo):", round(r2_score(y_test, y_pred), 4))
print(
    "Spearman rank corr:",
    round(spearmanr(y_test, y_pred).correlation, 4)
)

# ===================== SCORING FULL MARKET =====================
df["Score_raw"] = model.predict(X)

# ===================== GLOBAL SCALING =====================
df["Score_global"] = (df["Score_raw"] - df["Score_raw"].min()) / (
    df["Score_raw"].max() - df["Score_raw"].min()
)

# ===================== SAFE INDUSTRY SCALING =====================
def minmax_safe(x):
    if x.max() == x.min():
        return pd.Series([0.5] * len(x), index=x.index)
    return (x - x.min()) / (x.max() - x.min())

df["Score_industry"] = (
    df.groupby("industry")["Score_raw"]
      .transform(minmax_safe)
)

# ===================== RANKING THEO NGÀNH =====================
df = df.sort_values(
    ["industry", "Score_industry"],
    ascending=[True, False]
).reset_index(drop=True)

df["Rank_in_Industry"] = (
    df.groupby("industry")["Score_industry"]
      .rank(method="first", ascending=False)
      .astype(int)
)

# ===================== EXPORT =====================
cols = [
    "ticker",
    "industry",
    "Rank_in_Industry",
    "Score_industry",
    "Score_global",
    "ROA", "DE", "BV", "PB", "EPS", "Market_Cap"
]

df[cols].to_csv(
    "ket_qua_ranking_co_phieu_ml_industry_scaled.csv",
    index=False,
    encoding="utf-8-sig"
)

print("✅ Đã lưu: ket_qua_ranking_co_phieu_ml_industry_scaled.csv")
