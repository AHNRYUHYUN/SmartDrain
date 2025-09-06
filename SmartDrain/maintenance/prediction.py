# prediction.py
import os
import numpy as np
import joblib
from typing import List, Optional

from django.conf import settings
from accountapp.models import SensorLog  # 모델 경로 확인

MODEL_DIR = getattr(settings, "MODEL_DIR", "trained_models")


def load_model_and_scaler(drain_id: int):
    """
    Keras 3: compile=False 로 로드 (mse 직렬화 이슈 회피).
    모델/스케일러가 없으면 (None, None) 반환.
    """
    try:
        from tensorflow import keras  # 지연 임포트
    except Exception:
        return None, None

    model_path = os.path.join(MODEL_DIR, f"model_drain_{drain_id}.h5")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_drain_{drain_id}.pkl")

    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        return None, None

    try:
        model = keras.models.load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception:
        return None, None


# ---------- 데이터 준비 ----------
def _latest_seq(drain, seq_len: int, sensor_type: str = "초음파 센서"):
    """
    최신 seq_len개 시퀀스를 '그대로'(0 포함) 과거→현재 순서로 반환.
    반환: (np.ndarray[seq_len], first_ts) 또는 (None, None)
    """
    qs = (
        SensorLog.objects
        .filter(drain=drain, sensor_type=sensor_type)
        .order_by("-timestamp", "-id")
    )
    rows = list(qs[:seq_len])
    if not rows:
        return None, None

    # 최신→과거로 가져온 것을 과거→현재로 뒤집기
    values = [float(r.value) for r in reversed(rows)]
    seq = np.array(values, dtype=np.float32)
    return seq, rows[-1].timestamp


# ---------- 예측 ----------
def forecast_next_hours_safe(drain, lead_hours: int = 24) -> List[float]:
    """
    항상 '리스트'를 반환하는 안전 예측.
    - 시퀀스는 0 포함 (청소 마크도 그대로 학습/예측에 반영)
    - 모델/스케일러 없거나 차원 불일치/실패 시 → 마지막값 유지(나이브)
    """
    seq_len_cfg = int(getattr(settings, "MODEL_SEQ_LEN", 24))

    seq, _ = _latest_seq(drain, seq_len=seq_len_cfg)
    if seq is None or len(seq) == 0:
        return []  # 예측 불가

    last_v = float(seq[-1])
    naive = [last_v] * lead_hours

    model, scaler = load_model_and_scaler(drain.id)
    if model is None or scaler is None or not hasattr(model, "input_shape"):
        return naive

    # 기대 입력 차원: (batch, timesteps, features)
    try:
        _, exp_len, exp_feats = model.input_shape
    except Exception:
        return naive

    # feature 수가 1이 아니면(훈련 때 보조특성 썼던 모델) 여기선 맞출 수 없어 폴백
    if exp_feats != 1:
        return naive

    use_len = exp_len or seq_len_cfg

    # 길이가 부족하면 왼쪽 패딩으로 길이를 맞춰서라도 '모델'을 최대한 사용
    if len(seq) < use_len:
        pad = int(use_len - len(seq))
        pad_val = float(seq[0])
        seq_in = np.pad(seq, (pad, 0), constant_values=pad_val)
    else:
        seq_in = seq[-use_len:]

    # 스케일링
    X = seq_in.reshape(-1, 1)  # (timesteps, 1)
    try:
        if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != 1:
            return naive
        Xs = scaler.transform(X).reshape(1, use_len, 1)
    except Exception:
        return naive

    # 예측
    try:
        y = model.predict(Xs, verbose=0)[0]  # 보통 (lead_hours,)
        preds = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        preds = scaler.inverse_transform(preds).ravel().tolist()
        # 길이 보정(부족하면 마지막값으로 패딩, 길면 절단)
        if len(preds) < lead_hours:
            preds += [preds[-1]] * (lead_hours - len(preds))
        else:
            preds = preds[:lead_hours]
        return [float(v) for v in preds]
    except Exception:
        return naive


def forecast_next_value(drain) -> Optional[float]:
    """
    1-step(next)만 단일 float로 반환. 실패 시 None.
    (마지막 값이 0이어도 모델이 그대로 보고 다음을 예측)
    """
    preds = forecast_next_hours_safe(drain, lead_hours=1)
    if not preds:
        return None
    return float(preds[0])


# ---------- 하위 호환(언패킹 방지용 래퍼) ----------
forecast_next_hours = forecast_next_hours_safe  # alias

def forecast_next_hours_tuple(drain, lead_hours: int = 24, *args, **kwargs):
    """
    과거에 (preds, meta) 형태로 언패킹하던 코드를 위한 래퍼.
    항상 (리스트, 메타딕트) 튜플을 반환.
    """
    preds = forecast_next_hours_safe(drain, lead_hours=lead_hours)
    meta = {"len": len(preds), "source": "safe"}
    return preds, meta

def forecast_next_value_pair(drain, *args, **kwargs):
    """
    과거에 (next, meta)처럼 언패킹하던 코드를 위한 래퍼.
    항상 (float|None, None) 튜플을 반환.
    """
    v = forecast_next_value(drain)
    return v, None


__all__ = [
    "load_model_and_scaler",
    "forecast_next_hours_safe",
    "forecast_next_hours",        # alias
    "forecast_next_hours_tuple",  # tuple 래퍼
    "forecast_next_value",
    "forecast_next_value_pair",   # tuple 래퍼
]
