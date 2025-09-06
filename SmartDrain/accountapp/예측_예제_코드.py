import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# CSV 불러오기
df = pd.read_csv("C:/Users/AhnRyuHyun/Downloads/하수구_예측_학습데이터_100건.csv")

# 피처와 타겟 분리
X = df[['hour', 'weekday', 'motor_activations', 'rain_flag', 'is_night', 'weekend_flag', 'cluster']]
y = df['next_day_avg_trash_height']

# 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 현재 환경에서 pkl 저장
joblib.dump(model, "predictor.pkl")
