from tensorflow.keras.models import load_model
from django.conf import settings
import os

# ...
MODEL_DIR = os.path.join(getattr(settings, 'BASE_DIR', ''), "trained_models")  # 절대경로 권장
PLOT_SUBDIR = "plots"  # 모델 폴더 하위 plots 디렉터리


from rest_framework import viewsets
from .models import SensorLog,AlertLog
from .serializers import (
    SensorLogSerializer,

    AlertLogSerializer, TrashStatSerializer, DrainInfoSerializer
)
from .models import PredictionResult
from .serializers import PredictionResultSerializer
from django.http import JsonResponse
from .management.commands.update_trash_stats import Command
from django.shortcuts import render
from .models import SensorLog, TrashStat, DrainInfo
from django.utils import timezone
from datetime import datetime, timedelta
from django.db.models import Avg, Max

# predictor.pkl 로드
#model = joblib.load("C:/Users/AhnRyuHyun/PycharmProjects/pythonProject/SmartDrain/accountapp/predictor.pkl")
from rest_framework.response import Response
from rest_framework import status

from rest_framework import viewsets, filters
from rest_framework.permissions import AllowAny
from .models import RainLog
from .serializers import RainLogSerializer

class RainLogViewSet(viewsets.ModelViewSet):
    queryset = RainLog.objects.all()
    serializer_class = RainLogSerializer
    permission_classes = [AllowAny]  # 필요에 따라 IsAuthenticated 등으로 변경
    filter_backends = [filters.OrderingFilter]
    ordering = ['-timestamp']  # 기본 정렬

    def get_queryset(self):
        qs = super().get_queryset()
        # /api/rain-logs/?drain=1 식으로 특정 하수구 필터 가능
        drain_id = self.request.query_params.get('drain')
        if drain_id:
            qs = qs.filter(drain_id=drain_id)
        return qs
class SensorLogViewSet(viewsets.ModelViewSet):
    queryset = SensorLog.objects.all()
    serializer_class = SensorLogSerializer

    def get_queryset(self):
        queryset = super().get_queryset()
        name = self.request.query_params.get('name')
        sensor_type = self.request.query_params.get('sensor_type')

        # 이름으로 drain 필터링
        if name:
            queryset = queryset.filter(drain__name=name)

        # 센서 유형으로 필터링
        if sensor_type:
            queryset = queryset.filter(sensor_type=sensor_type)

        return queryset

    def post(self, request):
        serializer = SensorLogSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
class DrainInfoViewSet(viewsets.ModelViewSet):
    queryset = DrainInfo.objects.all()
    serializer_class = DrainInfoSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            print("❌ Validation Error:", serializer.errors)  # 서버 콘솔에 출력
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        self.perform_create(serializer)
        return Response(serializer.data, status=status.HTTP_201_CREATED)





#센서값 입력 코드   센서타입 다 분할하기
class SensorLogViewSet(viewsets.ModelViewSet):
    queryset = SensorLog.objects.all()
    serializer_class = SensorLogSerializer

    def get_queryset(self):
        queryset = super().get_queryset()
        name = self.request.query_params.get('name')
        sensor_type = self.request.query_params.get('sensor_type')

        # 이름으로 drain 필터링
        if name:
            queryset = queryset.filter(drain__name=name)

        # 센서 유형으로 필터링
        if sensor_type:
            queryset = queryset.filter(sensor_type=sensor_type)

        return queryset

    def post(self, request):
        serializer = SensorLogSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)




class AlertLogViewSet(viewsets.ModelViewSet):
    queryset = AlertLog.objects.all()
    serializer_class = AlertLogSerializer

from rest_framework import status, serializers

from .models import DrainInfo, SensorLog


# 응답 timestamp를 KST "YYYY-MM-DD HH:00:00"로 고정
class KSTHourOnlyField(serializers.DateTimeField):
    def to_representation(self, value):
        if value is None:
            return None
        kst = ZoneInfo("Asia/Seoul")
        if timezone.is_aware(value):
            value = value.astimezone(kst)
        elif getattr(settings, "USE_TZ", False):
            value = value.replace(tzinfo=dt_timezone.utc).astimezone(kst)
        value = value.replace(minute=0, second=0, microsecond=0)
        return value.strftime("%Y-%m-%d %H:%M:%S")


class SensorLogSerializer(serializers.ModelSerializer):
    drain = serializers.StringRelatedField()
    timestamp = KSTHourOnlyField()

    class Meta:
        model = SensorLog
        fields = ("id", "drain", "timestamp", "sensor_type", "value")


from datetime import timedelta, timezone as dt_timezone
from zoneinfo import ZoneInfo
from django.utils import timezone
from django.db.models import Q

from datetime import datetime, timedelta
from django.db.models import Q
from django.utils import timezone
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.response import Response

# 프로젝트 경로에 맞게 수정
from .models import DrainInfo, SensorLog
from .serializers import SensorLogSerializer


class SensorLogsByDrainInfoAPIView(APIView):
    """
    POST: region, sub_region, name, (선택) sensor_type
      - 조건에 맞는 SensorLog 중 '가장 최신 1건'만 반환
      - 최신 판단 기준: timestamp DESC, id DESC
      - 응답 timestamp 포맷은 Serializer에서 처리
    """

    def post(self, request):
        region = request.data.get('region')
        sub_region = request.data.get('sub_region')
        name = request.data.get('name')
        sensor_type = request.data.get('sensor_type')  # 선택

        if not (region and sub_region and name):
            return Response({'error': 'region, sub_region, name 모두 필요합니다.'},
                            status=status.HTTP_400_BAD_REQUEST)

        try:
            drain = DrainInfo.objects.get(region=region, sub_region=sub_region, name=name)
        except DrainInfo.DoesNotExist:
            return Response({'error': '해당 DrainInfo를 찾을 수 없습니다.'},
                            status=status.HTTP_404_NOT_FOUND)

        qs = SensorLog.objects.filter(drain=drain)
        if sensor_type:
            qs = qs.filter(sensor_type=sensor_type)
        # 필요 시 기본 센서 유형을 강제하려면 아래 주석 해제
        # else:
        #     qs = qs.filter(sensor_type="초음파 센서")

        # 가장 최신 1건
        log = qs.order_by('-timestamp', '-id').first()

        serializer = SensorLogSerializer([log] if log else [], many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


#쓰레기 평균 최고치 코드
class TrashStatViewSet(viewsets.ModelViewSet):
    queryset = TrashStat.objects.all()
    serializer_class = TrashStatSerializer
def update_trash_stats_view(request):
    if request.method == 'GET':
        Command().handle()
        return JsonResponse({'status': 'ok', 'message': '통계가 갱신되었습니다.'})
def update_trash_stats_form(request):
    result = None
    error = None
    drains = DrainInfo.objects.all()

    if request.method == 'POST':
        drain_id = request.POST.get('drain_id')
        date_str = request.POST.get('date')

        if not drain_id:
            error = "하수구를 선택해주세요."
        else:
            try:
                drain = DrainInfo.objects.get(id=drain_id)
                target_date = datetime.strptime(date_str, '%Y-%m-%d').date() if date_str else timezone.now().date() - timedelta(days=1)

                start = timezone.make_aware(datetime.combine(target_date, datetime.min.time()))
                end = timezone.make_aware(datetime.combine(target_date + timedelta(days=1), datetime.min.time()))

                logs = SensorLog.objects.filter(
                    sensor_type='초음파 센서',
                    drain=drain,
                    timestamp__gte=start,
                    timestamp__lt=end
                )

                if logs.exists():
                    avg_height = logs.aggregate(avg=Avg('value'))['avg']
                    max_height = logs.aggregate(max=Max('value'))['max']

                    stat, created = TrashStat.objects.update_or_create(
                        drain=drain,
                        date=target_date,
                        defaults={'avg_height': avg_height, 'max_height': max_height}
                    )
                    result = f"{target_date} - {drain}: 평균 {avg_height:.2f}cm / 최대 {max_height:.2f}cm → {'생성됨' if created else '업데이트됨'}"
                else:
                    result = f"{target_date} - {drain}: 초음파 센서 로그가 없습니다."
            except Exception as e:
                error = str(e)

    return render(request, 'accountapp/update_trash_stats_form.html', {
        'drains': drains,
        'result': result,
        'error': error
    })
class PredictionResultViewSet(viewsets.ModelViewSet):
    queryset = PredictionResult.objects.all()
    serializer_class = PredictionResultSerializer


class PredictDrainStatusView(APIView):
    def post(self, request):
        # 1. 요청 파라미터 확인
        region = request.data.get("region")
        sub_region = request.data.get("sub_region")
        name = request.data.get("name")
        days_ahead = int(request.data.get("days", 1))

        if not (region and sub_region and name):
            return Response({"error": "region, sub_region, name 값이 모두 필요합니다."}, status=400)

        # 2. DrainInfo 조회
        try:
            drain = DrainInfo.objects.get(region=region, sub_region=sub_region, name=name)
        except DrainInfo.DoesNotExist:
            return Response({"error": "해당 하수구 정보를 찾을 수 없습니다."}, status=404)

        drain_id = drain.id

        # 3. SensorLog 데이터 조회
        logs = SensorLog.objects.filter(
            drain=drain, sensor_type="초음파 센서"
        ).order_by("timestamp")

        if logs.count() < 10:
            return Response({"error": "예측에 필요한 시계열 데이터가 부족합니다."}, status=400)

        # 4. 시계열 데이터 전처리
        df = pd.DataFrame(list(logs.values("timestamp", "value")))
        df = df.dropna(subset=["timestamp", "value"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df.set_index("timestamp", inplace=True)
        df = df.resample("1D").mean().interpolate()

        # 5. 모델 및 스케일러 경로 설정
        model_path = f"trained_models/model_drain_{drain_id}.h5"
        scaler_path = f"trained_models/scaler_drain_{drain_id}.pkl"

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return Response({"error": "해당 하수구의 모델 또는 스케일러 파일이 존재하지 않습니다."}, status=400)

        # 6. 모델 및 스케일러 로딩 (오류 방지를 위해 compile=False)
        try:
            model = load_model(model_path, compile=False)
            scaler = joblib.load(scaler_path)
        except Exception as e:
            return Response({"error": f"모델 또는 스케일러 로딩 실패: {str(e)}"}, status=500)

        # 7. 입력 시퀀스 생성 및 스케일링
        scaled_data = scaler.transform(df[["value"]])
        last_seq = scaled_data[-10:]

        if last_seq.shape[0] < 10:
            return Response({"error": "예측을 위한 시퀀스가 부족합니다."}, status=400)

        # 8. 미래 예측 수행
        preds = []
        input_seq = last_seq.copy()
        for _ in range(days_ahead):
            pred = model.predict(input_seq[np.newaxis, :, :])[0][0]
            preds.append(pred)
            input_seq = np.vstack([input_seq[1:], [[pred]]])

        # 9. 예측값 역변환
        predicted_values = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        future_dates = [(df.index[-1] + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(days_ahead)]
        result = [{"date": d, "predicted_value": round(v, 2)} for d, v in zip(future_dates, predicted_values)]

        # 10. 결과 반환 및 DB 저장
        for date_str, predicted_value in zip(future_dates, predicted_values):
            predicted_value = round(predicted_value, 2)

            # 위험도 분류 기준 정의 (필요 시 조정 가능)
            if predicted_value < 20:
                risk_level = "low"
            elif predicted_value < 40:
                risk_level = "moderate"
            else:
                risk_level = "high"

            # 날짜 문자열 → datetime 객체로 변환
            predicted_date = datetime.strptime(date_str, "%Y-%m-%d")

            # PredictionResult 객체 생성 및 저장
            PredictionResult.objects.create(
                drain=drain,
                date=predicted_date,
                predicted_risk_level=risk_level,
                predicted_value=predicted_value
            )

        # 11. 클라이언트에 결과 반환
        return Response(result, status=200)

#모델 학습 코드
# -*- coding: utf-8 -*-
from pathlib import Path
import os, glob, logging
import numpy as np
import pandas as pd
import joblib

from django.conf import settings
from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ✅ 서버/헤드리스에서도 PNG 저장되도록
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 실제 앱 경로로 변경하세요
from accountapp.models import DrainInfo, SensorLog


import os
import glob
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from django.conf import settings
from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

# 프로젝트 모델 (경로 환경에 맞게 조정)
from .models import DrainInfo, SensorLog


def _maybe_enable_mixed_precision():
    """GPU가 있으면 mixed_float16 적용 (CPU만 있으면 제외)"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            return "mixed_float16"
        except Exception:
            pass
    return "float32"


def _maybe_enable_xla():
    """가능한 환경에서 XLA JIT 시도 (안되면 무시)."""
    try:
        tf.config.optimizer.set_jit(True)
        return True
    except Exception:
        return False


# -*- coding: utf-8 -*-
# TrainImprovedDrainModelsView: 정확도/로스만 JSON으로 리턴 + 시각화(base64) + (옵션) PNG 파일 저장
import os, io, json, base64, logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 서버 환경에서 GUI 없이 렌더
import matplotlib.pyplot as plt

import joblib
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LSTM, GRU, Conv1D, GlobalAveragePooling1D,
    LayerNormalization, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from django.conf import settings
from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

# 필요에 맞게 실제 경로로 수정하세요.
#   예) from drains.models import DrainInfo, SensorLog
from .models import DrainInfo, SensorLog


# === 런타임 가속 옵션 ===
def _maybe_enable_mixed_precision() -> str:
    """
    GPU가 있으면 mixed_float16 정책을 적용합니다. (TF 2.10 호환)
    """
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_global_policy(policy)
            return "mixed_float16"
    except Exception:
        pass
    return "float32"


def _maybe_enable_xla() -> bool:
    """
    가능한 경우 XLA JIT을 활성화합니다.
    """
    try:
        tf.config.optimizer.set_jit(True)
        return True
    except Exception:
        return False


# -*- coding: utf-8 -*-
from pathlib import Path
import os, glob, logging
import numpy as np
import pandas as pd
import joblib

from django.conf import settings
from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ✅ 서버/헤드리스에서도 PNG 저장되도록
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 실제 앱 경로로 변경하세요
from accountapp.models import DrainInfo, SensorLog

class TrainImprovedDrainModelsView(APIView):
    """
    - 초음파 전용 + cleaning_flag 포함 시계열 예측
    - 멀티모델(LSTM/GRU/CNN1D) 비교, 베스트 선정
    - 산출물: 발표용 PNG들 (정확도 강조 대시보드 + 학습진행 그래프 등)
    - JSON 파일 생성하지 않음
    """

    # ===== 출력/옵션 =====
    SHOW_PERFORMANCE_ONLY = True
    GENERATE_TEST_TIMELINE = False
    GENERATE_MODEL_COMPARISON = False
    GENERATE_THRESHOLD_FORECAST = False

    # ===== 경로/하이퍼파라미터 =====
    MODEL_DIR = Path(getattr(settings, "BASE_DIR", Path("."))) / "trained_models"

    SAVE_ONLY_IF_BETTER = True
    BEAT_RATIO = 0.95

    SEQ_LEN = 24
    SENSOR_HEIGHT = '초음파 센서'
    USE_CLEANING_FLAG = True

    MODEL_KINDS = ["lstm2", "gru2", "cnn1d"]
    LEARNING_RATE = 1e-3
    EPOCHS = 180
    BATCH_SIZE = 16
    PATIENCE_ES = 14
    PATIENCE_RLR = 7
    MIN_LR = 1e-5
    DROPOUT = 0.2

    THRESHOLD_MM = 25.0
    FORECAST_HORIZON = 72

    # ===== 내부 콜백: 에포크별 검증 지표 수집 =====
    class _ValMetricsCallback(tf.keras.callbacks.Callback):
        def __init__(self, X_va, y_va, scaler, tol_mm=2.0):
            super().__init__()
            self.X_va = X_va
            self.y_va = y_va
            self.scaler = scaler
            self.tol_mm = tol_mm
            self.history = {"val_loss": [], "val_rmse": [], "val_mae": [], "val_acc2": []}

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            self.history["val_loss"].append(float(logs.get("val_loss", np.nan)))
            y_pred_scaled = self.model(self.X_va, training=False).numpy()
            y_true = self.scaler.inverse_transform(self.y_va).flatten()
            y_pred = self.scaler.inverse_transform(y_pred_scaled).flatten()
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            mae  = float(np.mean(np.abs(y_true - y_pred)))
            acc2 = float(np.mean(np.abs(y_true - y_pred) <= self.tol_mm)) if len(y_true) else 0.0
            self.history["val_rmse"].append(rmse)
            self.history["val_mae"].append(mae)
            self.history["val_acc2"].append(acc2)

    # ===== 유틸 =====
    @staticmethod
    def _safe_mape(y_t, y_p):
        y_t = np.asarray(y_t, dtype=float); y_p = np.asarray(y_p, dtype=float)
        eps = 1e-8
        return float(np.mean(np.abs((y_t - y_p) / (np.abs(y_t) + eps))) * 100)

    @staticmethod
    def _smape(y_t, y_p):
        y_t = np.asarray(y_t, dtype=float); y_p = np.asarray(y_p, dtype=float)
        eps = 1e-8
        return float(np.mean(2.0 * np.abs(y_p - y_t) / (np.abs(y_t) + np.abs(y_p) + eps)) * 100)

    @staticmethod
    def _mase(y_true, y_pred, train_targets):
        y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
        tr = np.asarray(train_targets, dtype=float)
        diffs = np.abs(np.diff(tr[~np.isnan(tr)]))
        scale = np.mean(diffs) if len(diffs) > 0 else 1.0
        if scale < 1e-8: scale = 1.0
        return float(np.mean(np.abs(y_true - y_pred)) / scale)

    @staticmethod
    def _within_tolerance(y_true, y_pred, tol_mm=2.0):
        y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
        if len(y_true) == 0: return 0.0
        return float(np.mean(np.abs(y_true - y_pred) <= tol_mm))

    # ----- 피처 구성 -----
    def _build_features(self, df_h):
        values = df_h['value'].to_numpy(dtype=float)
        pos_vals = values[(~np.isnan(values)) & (values > 0)]
        scaler = MinMaxScaler()
        if len(pos_vals) >= 2 and (np.max(pos_vals) - np.min(pos_vals) > 1e-12):
            scaler.fit(pos_vals.reshape(-1, 1))
        else:
            scaler.fit(np.array([[1.0], [30.0]]))

        scaled_all = np.full_like(values, np.nan, dtype=float)
        mask_valid = (~np.isnan(values))
        scaled_all[mask_valid] = scaler.transform(values[mask_valid].reshape(-1, 1)).reshape(-1)

        feat_list = [scaled_all.reshape(-1, 1)]
        if self.USE_CLEANING_FLAG:
            cleaning_flag = (df_h['value'] == 0).astype(float).to_numpy().reshape(-1, 1)
            feat_list.append(cleaning_flag)

        feat_mat = np.concatenate(feat_list, axis=1)
        return feat_mat, values, scaler

    # ----- 사이클 내부 시퀀스 -----
    @staticmethod
    def _create_sequences_cycle(feat_mat, raw_height, seq_len=24):
        X, y = [], []
        raw = np.asarray(raw_height, dtype=float)
        feats = np.asarray(feat_mat, dtype=float)
        n = len(raw)

        height_scaled = feats[:, 0:1]
        reset_idx = np.where(raw == 0)[0]
        bounds = [-1] + reset_idx.tolist() + [n - 1]

        for b in range(len(bounds) - 1):
            start = bounds[b] + 1
            end = bounds[b + 1]
            seg_len = end - start + 1
            if seg_len < seq_len + 1: continue
            fseg = feats[start:end + 1]
            rseg = raw[start:end + 1]
            for i in range(seg_len - seq_len):
                wr = rseg[i:i + seq_len + 1]
                wf = fseg[i:i + seq_len + 1]
                if np.any((wr == 0) | np.isnan(wr)) or np.any(np.isnan(wf)):
                    continue
                X.append(wf[:seq_len])
                y.append(height_scaled[start + i + seq_len, 0])

        if not X:
            return np.empty((0, seq_len, feats.shape[1])), np.empty((0, 1))
        return np.array(X), np.array(y).reshape(-1, 1)

    # ----- 모델 빌더 -----
    def _build_model(self, kind, input_shape):
        model = Sequential()
        model.add(Input(shape=input_shape))
        if kind == "lstm2":
            model.add(LSTM(64, return_sequences=True))
            model.add(Dropout(self.DROPOUT))
            model.add(LSTM(32, return_sequences=False))
            model.add(Dropout(self.DROPOUT))
            model.add(Dense(16, activation="relu"))
            model.add(Dense(1, activation="linear"))
        elif kind == "gru2":
            model.add(GRU(64, return_sequences=True))
            model.add(Dropout(self.DROPOUT))
            model.add(GRU(32, return_sequences=False))
            model.add(Dropout(self.DROPOUT))
            model.add(Dense(16, activation="relu"))
            model.add(Dense(1, activation="linear"))
        elif kind == "cnn1d":
            model.add(Conv1D(64, kernel_size=3, padding="causal", activation="relu"))
            model.add(Dropout(self.DROPOUT))
            model.add(Conv1D(32, kernel_size=3, padding="causal", activation="relu"))
            model.add(GlobalAveragePooling1D())
            model.add(Dense(16, activation="relu"))
            model.add(Dense(1, activation="linear"))
        else:
            raise ValueError(f"unknown model kind: {kind}")
        model.compile(optimizer=Adam(learning_rate=self.LEARNING_RATE), loss="mse")
        return model

    # ----- 지표 계산 -----
    def _evaluate_metrics(self, scaler, y_tr, y_true_scaled, y_pred_scaled):
        y_true = scaler.inverse_transform(y_true_scaled).flatten()
        y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mape_val = self._safe_mape(y_true, y_pred)
        smape_val = self._smape(y_true, y_pred)
        r2_val = float(r2_score(y_true, y_pred))
        mase_val = self._mase(y_true, y_pred, scaler.inverse_transform(y_tr).flatten())
        acc2mm = self._within_tolerance(y_true, y_pred, tol_mm=2.0)
        return {
            "mae": mae, "rmse": rmse, "mape": mape_val, "smape": smape_val,
            "r2": r2_val, "mase": mase_val, "acc_within_2mm": acc2mm,
            "y_true": y_true, "y_pred": y_pred
        }

    # ----- PNG 저장 공용 함수 -----
    def _save_figure_png(self, fig, out_path: Path):
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(str(out_path), format="png", dpi=130)
        plt.close(fig)
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError(f"failed to save figure: {out_path}")
        logging.info(f"[PNG SAVED] {out_path} ({out_path.stat().st_size} bytes)")
        return out_path

    # ----- 발표용 대시보드 (정확도 강조) -----
    def _save_metrics_dashboard(self, drain_id, metrics_per_model, best_kind):
        out = self.MODEL_DIR / f"metrics_dashboard_drain_{drain_id}.png"
        kinds = list(metrics_per_model.keys())
        cols = ["RMSE", "MAE", "MASE", "R²", "Acc≤2mm"]

        table_data, rmse, mae, mase, accp = [], [], [], [], []
        for k in kinds:
            v = metrics_per_model[k]
            table_data.append([
                f"{v['rmse']:.3f}",
                f"{v['mae']:.3f}",
                f"{v['mase']:.3f}",
                f"{v['r2']:.3f}",
                f"{(v.get('acc2',0.0)*100):.1f}%"
            ])
            rmse.append(v["rmse"]); mae.append(v["mae"]); mase.append(v["mase"]); accp.append(v.get("acc2",0.0)*100)

        fig = plt.figure(figsize=(9.6, 7.4), dpi=130)

        # 상단 테이블
        ax1 = fig.add_axes([0.05, 0.55, 0.9, 0.4]); ax1.axis('off')
        the_table = ax1.table(
            cellText=table_data,
            rowLabels=[f"{k} {'★' if k==best_kind else ''}" for k in kinds],
            colLabels=cols, loc='center', cellLoc='center'
        )
        the_table.auto_set_font_size(False); the_table.set_fontsize(10); the_table.scale(1.0, 1.35)
        ax1.set_title(f"드레인 {drain_id} — 모델별 성능 요약 (정확도 강조)", fontsize=14, weight="bold")

        # 하단 막대
        ax2 = fig.add_axes([0.08, 0.12, 0.84, 0.36])
        x = np.arange(len(kinds)); w = 0.2
        ax2.bar(x - 1.5*w, rmse, width=w, label="RMSE(↓)")
        ax2.bar(x - 0.5*w, mae,  width=w, label="MAE(↓)")
        ax2.bar(x + 0.5*w, mase, width=w, label="MASE(↓)")
        ax2.bar(x + 1.5*w, accp, width=w, label="정확도 Acc≤2mm % (↑)")
        ax2.set_xticks(x); ax2.set_xticklabels(kinds)
        ax2.grid(axis="y", alpha=0.25); ax2.legend(loc="best", framealpha=0.9)
        ax2.set_ylabel("값(오차는 낮을수록, 정확도는 높을수록 좋음)")
        ax2.set_title("핵심 지표 비교", fontsize=13)

        # 베스트 강조
        best_acc = metrics_per_model[best_kind].get("acc2", 0.0) * 100
        ax2.text(0.98, 0.98, f"Best {best_kind}\n정확도 {best_acc:.1f}%",
                 transform=ax2.transAxes, ha="right", va="top",
                 fontsize=12, weight="bold", color="tab:blue",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, lw=0.6))
        try:
            best_idx = kinds.index(best_kind)
            ax2.axvline(best_idx, ymin=0, ymax=1, linestyle='--', alpha=0.35)
        except Exception:
            pass

        return self._save_figure_png(fig, out)

    # ----- 학습진행 그래프(요청 이미지 느낌) -----
    def _save_training_progress(self, drain_id, kind, mon_history):
        out = self.MODEL_DIR / f"training_progress_drain_{drain_id}_{kind}.png"
        epochs = np.arange(1, len(mon_history["val_loss"]) + 1)

        fig = plt.figure(figsize=(10.5, 7.0), dpi=130)

        # 상단 대패널: RMSE & MAE
        ax_top = fig.add_axes([0.08, 0.58, 0.84, 0.34])
        ax_top.plot(epochs, mon_history["val_rmse"], linewidth=2.2, label="val RMSE")
        ax_top.plot(epochs, mon_history["val_mae"],  linewidth=2.0, alpha=0.7, label="val MAE")
        ax_top.set_title("Model Performance", fontsize=13, weight="bold")
        ax_top.set_xlabel("Epochs"); ax_top.set_ylabel("Error")
        ax_top.grid(alpha=0.25); ax_top.legend(loc="best", framealpha=0.9)

        # 좌하: Loss
        ax1 = fig.add_axes([0.08, 0.10, 0.26, 0.29])
        ax1.plot(epochs, mon_history["val_loss"], linewidth=2.0)
        ax1.set_title("Val Loss"); ax1.set_xlabel("Epochs"); ax1.set_ylabel("Loss"); ax1.grid(alpha=0.25)

        # 중하: MAE
        ax2 = fig.add_axes([0.37, 0.10, 0.26, 0.29])
        ax2.plot(epochs, mon_history["val_mae"], linewidth=2.0)
        ax2.set_title("Val MAE"); ax2.set_xlabel("Epochs"); ax2.set_ylabel("MAE"); ax2.grid(alpha=0.25)

        # 우하: Acc(≤2mm)
        ax3 = fig.add_axes([0.66, 0.10, 0.26, 0.29])
        acc_percent = [v * 100.0 for v in mon_history["val_acc2"]]
        ax3.plot(epochs, acc_percent, linewidth=2.0)
        ax3.set_title("Accuracy (≤2mm)"); ax3.set_xlabel("Epochs"); ax3.set_ylabel("%"); ax3.grid(alpha=0.25)

        fig.suptitle(f"Training Progress — Drain {drain_id} / {kind}", y=0.98, fontsize=14, weight="bold")
        self._save_figure_png(fig, out)
        return str(out)

    # ----- (옵션) 테스트 플롯 -----
    def _save_test_plot(self, drain_id, y_true, y_pred, rmse, mae, mase):
        out = self.MODEL_DIR / f"model_drain_{drain_id}_test.png"
        x = np.arange(len(y_true))
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=130)
        y_line = y_true.copy()
        for i in range(len(y_line) - 1):
            if (y_true[i + 1] - y_true[i]) < 0:
                y_line[i + 1] = np.nan
        ax.plot(x, y_line, linewidth=1.8, label="실측(상승)")
        ax.fill_between(x, 0, y_line, where=~np.isnan(y_line), alpha=0.18, step=None, label="_nolegend_")
        ax.plot(x, y_true, linestyle='None', marker='o', markersize=3, label="_nolegend_")

        dy = np.diff(y_true)
        drop_idx = np.where(dy < 0)[0] + 1
        if len(drop_idx) > 0:
            for i in drop_idx:
                ax.vlines(i, 0, y_true[i], linestyles='dashed', linewidth=1.0, alpha=0.6)
            ax.plot(drop_idx, y_true[drop_idx], linestyle='None', marker='v', markersize=5, alpha=0.9, label="리셋")

        ax.plot(x, y_pred, linestyle='--', linewidth=1.6, marker='o', markersize=2.5, label="예측", alpha=0.95)
        ax.set_title(f"드레인 {drain_id} — 테스트: 실측/예측 & 리셋 이벤트")
        ax.set_xlabel("샘플(시간 순)"); ax.set_ylabel("높이 (mm)")
        ax.grid(alpha=0.25); ax.legend(loc="best", framealpha=0.9)

        pad = max(1.0, (float(np.nanmax([y_true.max(), y_pred.max()])) - float(np.nanmin([y_true.min(), y_pred.min()]))) * 0.08)
        ax.set_ylim(float(np.nanmin([y_true.min(), y_pred.min()])) - pad,
                    float(np.nanmax([y_true.max(), y_pred.max()])) + pad)

        txt = (f"RMSE {rmse:.3f}\nMAE  {mae:.3f}\nMASE {mase:.3f}")
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, linewidth=0.5))

        return self._save_figure_png(fig, out)

    # ----- (옵션) 모델 비교 플롯 -----
    def _save_model_comparison_plot(self, drain_id, metrics_per_model):
        out = self.MODEL_DIR / f"models_comparison_drain_{drain_id}.png"
        kinds = list(metrics_per_model.keys())
        rmse = [metrics_per_model[k]["rmse"] for k in kinds]
        mae  = [metrics_per_model[k]["mae"]  for k in kinds]
        mase = [metrics_per_model[k]["mase"] for k in kinds]
        r2   = [metrics_per_model[k]["r2"] for k in kinds]

        fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=130)
        x = np.arange(len(kinds)); w = 0.2
        ax.bar(x - 1.5*w, rmse, width=w, label="RMSE")
        ax.bar(x - 0.5*w, mae,  width=w, label="MAE")
        ax.bar(x + 0.5*w, mase, width=w, label="MASE")
        ax.bar(x + 1.5*w, r2,   width=w, label="R²")
        ax.set_xticks(x); ax.set_xticklabels(kinds)
        ax.set_title(f"드레인 {drain_id} — 모델 비교 지표")
        ax.grid(axis="y", alpha=0.25); ax.legend()
        return self._save_figure_png(fig, out)

    # ----- (옵션) 롤링 예측 & 임계 도달 시점 -----
    def _rollout_and_threshold(self, model, latest_block_feats, scaler, threshold_mm, horizon):
        seq = latest_block_feats.astype('float32').copy()
        preds_scaled = []
        for _ in range(horizon):
            x_in = seq[np.newaxis, :, :].astype('float32')
            y_next_scaled = model(x_in, training=False).numpy()
            preds_scaled.append(float(y_next_scaled.item()))
            nf = seq[-1].copy()
            nf[0] = float(y_next_scaled.item())
            if len(nf) > 1: nf[1:] = 0.0
            seq = np.vstack([seq[1:], nf])

        forecast_mm = self._inverse_transform_column(scaler, np.array(preds_scaled, dtype='float32').reshape(-1,1)).flatten().tolist()
        reach_index = next((i for i, v in enumerate(forecast_mm) if v >= threshold_mm), None)
        return {"forecast_mm": forecast_mm, "reach_index": reach_index}

    @staticmethod
    def _inverse_transform_column(scaler, arr2d):
        # sklearn MinMaxScaler inverse_transform는 2D 입력 기대
        return scaler.inverse_transform(arr2d)

    # ===== 메인 =====
    def post(self, request):
        # 폴더/권한/청소(JSON 사용 안 함)
        preflight = {"model_dir": str(self.MODEL_DIR), "ok": True, "error": None}
        try:
            self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            # 혹시 남아있던 summary_drain_*.json 정리
            for p in glob.glob(str(self.MODEL_DIR / "summary_drain_*.json")):
                try: os.remove(p)
                except: pass
            # 쓰기 확인
            test_path = self.MODEL_DIR / "._w.tmp"
            with open(test_path, "w", encoding="utf-8") as f: f.write("ok")
            os.remove(test_path)
        except Exception as e:
            preflight["ok"] = False; preflight["error"] = repr(e)

        drains = DrainInfo.objects.all()
        results = []

        for drain in drains:
            try:
                # ---- 데이터 ----
                qs = SensorLog.objects.filter(drain=drain, sensor_type=self.SENSOR_HEIGHT).order_by('timestamp')
                if qs.count() < 50:
                    results.append({"drain": drain.id, "status": "데이터 부족"}); continue

                df = pd.DataFrame.from_records(qs.values('timestamp', 'value'))
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df = df[(df['value'] >= 0) & (df['value'] <= 30)]
                df = df.resample('1h').median()  # 소문자 'h'

                # 세그 내부 1칸 결측만 ffill, 리셋은 0 유지
                v = df['value'].copy()
                is_reset = (v == 0)
                seg_id = is_reset.cumsum()
                pos_series = v.mask(is_reset)
                pos_filled = pos_series.groupby(seg_id).transform(lambda s: s.ffill(limit=1))
                df['value'] = pos_filled.where(~is_reset, 0.0)

                if len(df) < self.SEQ_LEN + 30:
                    results.append({"drain": drain.id, "status": "유효 샘플 부족"}); continue

                # ---- 피처/시퀀스 ----
                feat_mat, values_raw, scaler = self._build_features(df)
                X, y_next_scaled = self._create_sequences_cycle(feat_mat, values_raw, seq_len=self.SEQ_LEN)
                if len(X) < 30:
                    results.append({"drain": drain.id, "status": "시퀀스 부족(사이클 내)"}); continue

                # split
                n = len(X)
                i_tr = int(n * 0.70)
                i_va = int(n * 0.85)
                X_tr, y_tr = X[:i_tr], y_next_scaled[:i_tr]
                X_va, y_va = X[i_tr:i_va], y_next_scaled[i_tr:i_va]
                X_te, y_te = X[i_va:],    y_next_scaled[i_va:]
                if len(X_va) < 10 or len(X_te) < 10:
                    results.append({"drain": drain.id, "status": "검증/테스트 부족"}); continue

                # dtype 고정(retracing 완화)
                X_tr = X_tr.astype('float32'); y_tr = y_tr.astype('float32')
                X_va = X_va.astype('float32'); y_va = y_va.astype('float32')
                X_te = X_te.astype('float32'); y_te = y_te.astype('float32')

                # ---- 학습/검증 ----
                metrics_map = {}
                model_objs = {}
                per_model_progress = {}

                for kind in self.MODEL_KINDS:
                    model = self._build_model(kind, input_shape=(X.shape[1], X.shape[2]))
                    es = EarlyStopping(monitor='val_loss', patience=self.PATIENCE_ES, restore_best_weights=True)
                    rl = ReduceLROnPlateau(monitor='val_loss', patience=self.PATIENCE_RLR, factor=0.5,
                                           min_lr=self.MIN_LR, verbose=0)
                    mon = self._ValMetricsCallback(X_va, y_va, scaler, tol_mm=2.0)

                    model.fit(
                        X_tr, y_tr,
                        validation_data=(X_va, y_va),
                        epochs=self.EPOCHS, batch_size=self.BATCH_SIZE,
                        callbacks=[es, rl, mon],
                        verbose=0
                    )

                    # 검증 성능(선정/표시)
                    y_va_pred_scaled = model(X_va, training=False).numpy()
                    m_va = self._evaluate_metrics(scaler, y_tr, y_va, y_va_pred_scaled)
                    metrics_map[kind] = {
                        "val_mae": m_va["mae"], "val_rmse": m_va["rmse"],
                        "val_mase": m_va["mase"], "val_r2": m_va["r2"],
                        "val_acc2": m_va.get("acc_within_2mm", 0.0)
                    }
                    model_objs[kind] = model

                    # 학습진행 그래프 저장
                    try:
                        per_model_progress[kind] = self._save_training_progress(drain.id, kind, mon.history)
                    except Exception as e:
                        logging.exception(f"training progress save failed: drain={drain.id} kind={kind} err={e!r}")
                        per_model_progress[kind] = None

                # 베스트 선정 (val MASE -> val RMSE)
                def sort_key(k): return (metrics_map[k]["val_mase"], metrics_map[k]["val_rmse"])
                best_kind = sorted(self.MODEL_KINDS, key=sort_key)[0]
                best_model = model_objs[best_kind]

                # ---- 테스트 성능(베스트) ----
                y_te_pred_scaled = best_model(X_te, training=False).numpy()
                m_te = self._evaluate_metrics(scaler, y_tr, y_te, y_te_pred_scaled)
                y_true = m_te["y_true"]; y_pred = m_te["y_pred"]

                # 저장 여부(나이브 대비)
                naive_pred = scaler.inverse_transform(X_te[:, -1, 0:1]).flatten()
                naive_rmse = float(np.sqrt(mean_squared_error(y_true, naive_pred)))
                naive_mae  = float(mean_absolute_error(y_true, naive_pred))
                beats = (m_te["mase"] <= 1.0 * self.BEAT_RATIO) or \
                        ((m_te["rmse"] <= naive_rmse * self.BEAT_RATIO) and (m_te["mae"] <= naive_mae * self.BEAT_RATIO))

                # 모델/스케일러 저장(.keras)
                model_path  = self.MODEL_DIR / f"model_drain_{drain.id}.keras"
                scaler_path = self.MODEL_DIR / f"scaler_drain_{drain.id}.pkl"
                status_msg = "saved" if ((not self.SAVE_ONLY_IF_BETTER) or beats) else "skipped (baseline better/close)"
                if status_msg == "saved":
                    if model_path.exists(): model_path.unlink()
                    if scaler_path.exists(): scaler_path.unlink()
                    best_model.save(str(model_path))
                    joblib.dump(scaler, str(scaler_path))

                # ---- 대시보드 입력(정확도 포함) ----
                metrics_for_dashboard = {}
                for k in self.MODEL_KINDS:
                    if k == best_kind:
                        metrics_for_dashboard[k] = {
                            "rmse": m_te["rmse"], "mae": m_te["mae"],
                            "mase": m_te["mase"], "r2": m_te["r2"],
                            "acc2": m_te.get("acc_within_2mm", 0.0)
                        }
                    else:
                        metrics_for_dashboard[k] = {
                            "rmse": metrics_map[k]["val_rmse"], "mae": metrics_map[k]["val_mae"],
                            "mase": metrics_map[k]["val_mase"], "r2": metrics_map[k]["val_r2"],
                            "acc2": metrics_map[k]["val_acc2"]
                        }

                # ---- PNG 저장 ----
                md_path = str(self._save_metrics_dashboard(drain.id, metrics_for_dashboard, best_kind))
                test_plot_path = comp_plot_path = forecast_plot_path = None

                if not self.SHOW_PERFORMANCE_ONLY:
                    if self.GENERATE_TEST_TIMELINE:
                        test_plot_path = str(self._save_test_plot(
                            drain.id, y_true, y_pred,
                            m_te["rmse"], m_te["mae"], m_te["mase"]
                        ))
                    if self.GENERATE_MODEL_COMPARISON:
                        comp_plot_path = str(self._save_model_comparison_plot(
                            drain.id,
                            {k: {"rmse": metrics_for_dashboard[k]["rmse"],
                                 "mae": metrics_for_dashboard[k]["mae"],
                                 "mase": metrics_for_dashboard[k]["mase"],
                                 "r2": metrics_for_dashboard[k]["r2"]} for k in self.MODEL_KINDS}
                        ))
                    if self.GENERATE_THRESHOLD_FORECAST:
                        last_block = feat_mat[-self.SEQ_LEN:, :]
                        last_raw = values_raw[-self.SEQ_LEN:]
                        if (not np.any(np.isnan(last_block))) and (not np.any(last_raw == 0)):
                            roll = self._rollout_and_threshold(
                                best_model, last_block, scaler,
                                threshold_mm=self.THRESHOLD_MM, horizon=self.FORECAST_HORIZON
                            )
                            last_val_mm = float(values_raw[-1])
                            forecast_plot_path = str(self._save_threshold_plot(
                                drain.id, last_val_mm, roll["forecast_mm"], roll["reach_index"]
                            ))

                # ---- 응답(JSON은 파일경로/지표만; 별도 요약 JSON 저장 없음) ----
                results.append({
                    "drain": drain.id,
                    "status": status_msg,
                    "best_model": best_kind,
                    "beats_naive": bool(beats),
                    "seq_len": self.SEQ_LEN,
                    "count_train": int(len(X_tr)),
                    "count_val": int(len(X_va)),
                    "count_test": int(len(X_te)),
                    "test_metrics": {k: m_te[k] for k in ["mae","rmse","mape","smape","r2","mase","acc_within_2mm"]},
                    "plots": {
                        "metrics_dashboard": md_path,
                        "training_progress": per_model_progress.get(best_kind),  # 베스트 모델 진행 그래프
                        "test_plot": test_plot_path if not self.SHOW_PERFORMANCE_ONLY else None,
                        "model_comparison": comp_plot_path if not self.SHOW_PERFORMANCE_ONLY else None,
                        "threshold_forecast": forecast_plot_path if not self.SHOW_PERFORMANCE_ONLY else None
                    },
                })

            except Exception as e:
                logging.exception(f"Unexpected error for drain {getattr(drain,'id','?')}")
                results.append({"drain": getattr(drain,'id',None), "status": f"error: {e!r}"})

        # 혹시 외부 코드가 만든 summary JSON이 있다면 마무리 청소(선택)
        for p in glob.glob(str(self.MODEL_DIR / "summary_drain_*.json")):
            try: os.remove(p)
            except: pass

        return Response({
            "preflight": preflight,
            "model_dir": str(self.MODEL_DIR.resolve()),
            "processed": len(drains),
            "results": results,
            "timestamp": timezone.now().isoformat(),
        }, status=200)
