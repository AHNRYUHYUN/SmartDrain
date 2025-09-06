# maintenance/forecast_util.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from datetime import timedelta
from django.utils import timezone
from django.conf import settings

# 센서 로그 위치가 accountapp에 있다면 그대로 두세요.
# 다른 앱이면 경로만 바꾸세요.
from accountapp.models import SensorLog

# 모델 저장 경로 (프로젝트 루트 기준 trained_models)
MODEL_DIR = os.path.join(getattr(settings, "BASE_DIR", Path(".")), "trained_models")

def load_model_and_scaler(drain_id: int):
    """
    Keras 3: compile=False로 로드해 'mse' 역직렬화 문제 방지
    """
    from tensorflow import keras

    model_path  = os.path.join(MODEL_DIR, f"model_drain_{drain_id}.h5")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_drain_{drain_id}.pkl")
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        return None, None

    model = keras.models.load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    return model, scaler


def prepare_latest_sequence(drain, seq_len: int):
    """
    학습과 동일하게: 1H 리샘플(중앙값), 0은 리셋.
    '최근 연속 양수(>0)' 구간에서 seq_len개 확보.
    반환: (latest_values: (seq_len,), last_index: pd.Timestamp) or (None, None)
    """
    qs = SensorLog.objects.filter(drain=drain, sensor_type='초음파 센서').order_by('timestamp')
    if qs.count() < seq_len:
        return None, None

    df = pd.DataFrame.from_records(qs.values('timestamp', 'value'))
    if df.empty:
        return None, None

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # 유효 범위 필터(학습과 동일)
    df = df[(df['value'] >= 0) & (df['value'] <= 30)]
    # 1H 중앙값 리샘플
    df = df.resample('1H').median()

    if df.empty or 'value' not in df.columns:
        return None, None

    v = df['value'].to_numpy(dtype=float)

    # 최근 '연속 양수(>0)' run 길이 계산
    pos_mask = (v > 0) & ~np.isnan(v)
    if not np.any(pos_mask):
        return None, None

    run_len = 0
    for x in pos_mask[::-1]:
        if x:
            run_len += 1
        else:
            break

    if run_len < seq_len:
        return None, None

    end_idx = len(v)
    latest_slice = slice(end_idx - seq_len, end_idx)
    latest_vals = v[latest_slice]

    last_index = df.index.to_list()[end_idx - 1]
    return latest_vals, last_index


def forecast_next_hours(drain, horizon_hours=None, seq_len=None):
    """
    recursive one-step 예측.
    - 입력 채널 수(F)는 모델에서 자동 감지 (1채널: height_scaled, 2+채널: cleaning_flag 등 추가)
    - 기본 예측 길이: 72시간(3일)
    반환: [(dt, yhat_mm), ...]
    """
    # 기본값: settings 우선, 없으면 72/24 사용
    if horizon_hours is None:
        horizon_hours = int(getattr(settings, "PREDICTION_HORIZON_HOURS", 72))
    if seq_len is None:
        seq_len = int(getattr(settings, "MODEL_SEQ_LEN", 24))

    model, scaler = load_model_and_scaler(drain.id)
    if (model is None) or (scaler is None):
        return []

    latest, last_index = prepare_latest_sequence(drain, seq_len)
    if latest is None:
        return []

    # height만 스케일링
    height_scaled = scaler.transform(latest.reshape(-1, 1)).reshape(-1)  # (seq_len,)

    # 모델 입력 피처 수(F) 감지
    in_shape = getattr(model, "input_shape", None)
    feat_dim = 1 if in_shape is None else int(in_shape[-1])

    # 초기 윈도우 구성
    if feat_dim == 1:
        window = height_scaled.reshape(seq_len, 1)
    else:
        # 첫 채널: height_scaled, 나머지 채널: 0 (예측창에서는 cleaning_flag=0 가정)
        window = np.zeros((seq_len, feat_dim), dtype=float)
        window[:, 0] = height_scaled

    preds = []
    cur_time = last_index

    for _ in range(horizon_hours):
        yhat_scaled = float(model.predict(window.reshape(1, seq_len, feat_dim), verbose=0).reshape(-1)[0])
        yhat = float(scaler.inverse_transform(np.array([[yhat_scaled]])).reshape(-1)[0])

        cur_time = cur_time + timedelta(hours=1)
        preds.append((cur_time, yhat))

        if feat_dim == 1:
            window = np.vstack([window[1:], [[yhat_scaled]]])
        else:
            next_row = np.zeros((feat_dim,), dtype=float)
            next_row[0] = yhat_scaled
            window = np.vstack([window[1:], next_row[None, :]])

    return preds
