from tensorflow.keras.models import load_model
from django.conf import settings
import os

# ...
MODEL_DIR = os.path.join(getattr(settings, 'BASE_DIR', ''), "trained_models")  # 절대경로 권장
PLOT_SUBDIR = "plots"  # 모델 폴더 하위 plots 디렉터리


from rest_framework import viewsets
from .models import SensorLog, MotorLog, AlertLog, TrashStat, PredictionResult, DrainInfo
from .serializers import (
    SensorLogSerializer, MotorLogSerializer,
    AlertLogSerializer, TrashStatSerializer, PredictionResultSerializer, DrainInfoSerializer
)
import joblib
from rest_framework.views import APIView
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


class MotorLogViewSet(viewsets.ModelViewSet):
    queryset = MotorLog.objects.all()
    serializer_class = MotorLogSerializer



class AlertLogViewSet(viewsets.ModelViewSet):
    queryset = AlertLog.objects.all()
    serializer_class = AlertLogSerializer


#센서 값 반환해주는 코드
# views.py

from datetime import timedelta, timezone as dt_timezone
from zoneinfo import ZoneInfo
from django.conf import settings
from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response
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


# ===== Imports =====
import os

import matplotlib
matplotlib.use("Agg")


# ===== Imports =====
import logging


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 프로젝트 모델
from .models import DrainInfo, SensorLog

logger = logging.getLogger(__name__)


# ===== Imports =====
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 프로젝트 모델
from .models import DrainInfo, SensorLog

logger = logging.getLogger(__name__)


# ===== Imports =====
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .models import DrainInfo, SensorLog

logger = logging.getLogger(__name__)


import logging


logger = logging.getLogger(__name__)

# ===== Imports =====
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from django.utils import timezone
from django.conf import settings

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .models import DrainInfo, SensorLog

logger = logging.getLogger(__name__)


# -*- coding: utf-8 -*-
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from django.conf import settings
from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 모델/데이터 의존 모델은 프로젝트에 맞게 import 하세요.
# 예시)
from accountapp.models import DrainInfo
from accountapp.models import SensorLog  # 실제 앱 경로에 맞게 수정

# 텐서플로/케라스
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, Conv1D, GlobalAveragePooling1D, InputLayer
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class TrainImprovedDrainModelsView(APIView):
    """
    초음파 전용 + cleaning_flag 포함 LSTM/GRU/CNN 예측 (멀티모델 비교)
    - 리셋(값=0)으로 사이클 분할, 사이클 내부에서만 시퀀스 생성
    - 입력: height_scaled, cleaning_flag(현재 시점이 리셋=1), [추후 외부변수 확장 지점]
    - 모델 후보: LSTM(2층), GRU(2층), 1D-CNN(Conv1D+GAP)
    - 손실: MSE(Adam)
    - 산출물(기본): metrics_dashboard_drain_{id}.png (성능 지표 대시보드)
    - 옵션: 타임라인형 테스트 플롯, 모델비교 플롯, 임계도달 롤링 예측 플롯
    """

    # ===== 표시/출력 플래그 =====
    SHOW_PERFORMANCE_ONLY = True     # 성능 지표 대시보드만 출력
    GENERATE_TEST_TIMELINE = False   # 테스트 구간 시간축 그래프 (실측/예측 + 리셋 이벤트)
    GENERATE_MODEL_COMPARISON = False
    GENERATE_THRESHOLD_FORECAST = False

    # ===== 하이퍼파라미터 / 경로 =====
    SAVE_ONLY_IF_BETTER = True
    BEAT_RATIO = 0.95

    SEQ_LEN = 24  # 24시간 컨텍스트
    SENSOR_HEIGHT = '초음파 센서'
    USE_CLEANING_FLAG = True  # 리셋(=0) 시점 표시 feature 사용
    MODEL_DIR = Path(getattr(settings, "BASE_DIR", Path("."))) / "trained_models"

    # 멀티모델
    MODEL_KINDS = ["lstm2", "gru2", "cnn1d"]
    LEARNING_RATE = 1e-3
    EPOCHS = 180
    BATCH_SIZE = 16
    PATIENCE_ES = 14
    PATIENCE_RLR = 7
    MIN_LR = 1e-5
    DROPOUT = 0.2

    # 임계 도달 예측(롤링)
    THRESHOLD_MM = 25.0
    FORECAST_HORIZON = 72  # 시간 스텝 수(=1H 기준 72시간)

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
        """
        df_h: index=datetime(1H), column 'value' (mm), 0 = reset
        반환: feat_mat (N,F), values_raw (N,), scaler
              F = 1(height_scaled) + [1(cleaning_flag)]
        """
        values = df_h['value'].to_numpy(dtype=float)

        # 스케일러(양수만)
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

        # TODO: 외부 변수(강수량 등) 추가 시 여기서 병합하여 feat_list.append(...)
        feat_mat = np.concatenate(feat_list, axis=1)  # (N,F)
        return feat_mat, values, scaler

    # ----- 사이클 내부 시퀀스 -----
    @staticmethod
    def _create_sequences_cycle(feat_mat, raw_height, seq_len=24):
        """
        - 리셋(0) 경계를 기준으로 사이클 분할
        - 사이클 내부에서만 (seq_len+1) 윈도우 생성
        - NaN/0 포함 윈도우 제외
        반환: X:(M,seq_len,F), y:(M,1)  (y는 '다음 시점 height_scaled')
        """
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
            if seg_len < seq_len + 1:
                continue
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
        model = Sequential([InputLayer(input_shape=input_shape)])
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

    # ----- 성능 지표 대시보드 저장 -----
    def _save_metrics_dashboard(self, drain_id, metrics_per_model, best_kind):
        """
        metrics_per_model: dict[kind] = {"rmse":..., "mae":..., "mase":..., "r2":..., "acc2":... (선택)}
        """
        out = self.MODEL_DIR / f"metrics_dashboard_drain_{drain_id}.png"

        kinds = list(metrics_per_model.keys())
        cols = ["RMSE", "MAE", "MASE", "R²"]
        has_acc2 = any("acc2" in v for v in metrics_per_model.values())
        if has_acc2:
            cols.append("Acc≤2mm")

        # 테이블 데이터 구성 (소수 3자리)
        table_data = []
        for k in kinds:
            row = [
                f"{metrics_per_model[k]['rmse']:.3f}",
                f"{metrics_per_model[k]['mae']:.3f}",
                f"{metrics_per_model[k]['mase']:.3f}",
                f"{metrics_per_model[k]['r2']:.3f}",
            ]
            if has_acc2:
                row.append(f"{metrics_per_model[k]['acc2']*100:.1f}%")
            table_data.append(row)

        # 바차트용 배열
        rmse = [metrics_per_model[k]["rmse"] for k in kinds]
        mae  = [metrics_per_model[k]["mae"]  for k in kinds]
        mase = [metrics_per_model[k]["mase"] for k in kinds]

        fig = plt.figure(figsize=(8.5, 7.0), dpi=120)

        # ---- 상단: 테이블 ----
        ax1 = fig.add_axes([0.05, 0.58, 0.9, 0.37])
        ax1.axis('off')
        the_table = ax1.table(
            cellText=table_data,
            rowLabels=[f"{k} {'★' if k==best_kind else ''}" for k in kinds],
            colLabels=cols,
            loc='center',
            cellLoc='center'
        )
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(9)
        the_table.scale(1, 1.3)
        ax1.set_title(f"드레인 {drain_id} — 모델별 성능 지표")

        # ---- 하단: 가로 바차트 : MASE, RMSE, MAE 비교 ----
        ax2 = fig.add_axes([0.08, 0.08, 0.84, 0.42])
        y = np.arange(len(kinds))
        h = 0.22
        ax2.barh(y + 0.24, mase, height=h, label="MASE")
        ax2.barh(y + 0.00, rmse, height=h, label="RMSE")
        ax2.barh(y - 0.24, mae,  height=h, label="MAE")
        ax2.set_yticks(y)
        ax2.set_yticklabels(kinds)
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.25)
        ax2.legend(loc="lower right", framealpha=0.9)
        ax2.set_xlabel("값이 낮을수록 좋음")
        ax2.set_title("핵심 지표 비교 (낮을수록 좋음)")

        # 베스트 모델 강조 라인
        try:
            best_idx = kinds.index(best_kind)
            ax2.axhline(best_idx, xmin=0.0, xmax=1.0, linewidth=1.0, linestyle='--', alpha=0.6)
        except Exception:
            pass

        fig.tight_layout()
        fig.savefig(str(out), format="png")
        plt.close(fig)
        return out

    # ----- (옵션) 테스트 플롯 저장 -----
    def _save_test_plot(self, drain_id, y_true, y_pred,
                        rmse, mae, mase):
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        file_path = self.MODEL_DIR / f"model_drain_{drain_id}_test.png"

        x = np.arange(len(y_true))
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=120)

        # 상승(쌓이는) 구간만 선으로 연결
        y_line = y_true.copy()
        for i in range(len(y_line) - 1):
            if (y_true[i + 1] - y_true[i]) < 0:
                y_line[i + 1] = np.nan
        ax.plot(x, y_line, linewidth=1.8, label="실측(상승)")
        ax.fill_between(x, 0, y_line, where=~np.isnan(y_line), alpha=0.20, step=None, label="_nolegend_")
        ax.plot(x, y_true, linestyle='None', marker='o', markersize=3, label="_nolegend_")

        # 리셋(하강) 이벤트
        dy = np.diff(y_true)
        drop_idx = np.where(dy < 0)[0] + 1
        if len(drop_idx) > 0:
            for i in drop_idx:
                ax.vlines(i, 0, y_true[i], linestyles='dashed', linewidth=1.0, alpha=0.6)
            ax.plot(drop_idx, y_true[drop_idx], linestyle='None', marker='v', markersize=5, alpha=0.9, label="리셋")

        # 예측
        ax.plot(x, y_pred, linestyle='--', linewidth=1.6, marker='o', markersize=2.5, label="예측", alpha=0.95)

        ax.set_title(f"드레인 {drain_id} — 테스트: 실측/예측 & 리셋 이벤트")
        ax.set_xlabel("샘플(시간 순)")
        ax.set_ylabel("높이 (mm)")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", framealpha=0.9)

        ymin = float(np.nanmin([y_true.min(), y_pred.min()]))
        ymax = float(np.nanmax([y_true.max(), y_pred.max()]))
        pad = max(1.0, (ymax - ymin) * 0.08)
        ax.set_ylim(ymin - pad, ymax + pad)

        txt = (f"RMSE {rmse:.3f}\n"
               f"MAE  {mae:.3f}\n"
               f"MASE {mase:.3f}")
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, linewidth=0.5))

        fig.tight_layout()
        fig.savefig(str(file_path), format="png")
        plt.close(fig)
        return file_path

    # ----- (옵션) 모델 비교 플롯 -----
    def _save_model_comparison_plot(self, drain_id, metrics_per_model):
        out = self.MODEL_DIR / f"models_comparison_drain_{drain_id}.png"
        kinds = list(metrics_per_model.keys())
        rmse = [metrics_per_model[k]["rmse"] for k in kinds]
        mae  = [metrics_per_model[k]["mae"] for k in kinds]
        mase = [metrics_per_model[k]["mase"] for k in kinds]
        r2   = [metrics_per_model[k]["r2"] for k in kinds]

        fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=120)
        x = np.arange(len(kinds))
        w = 0.2
        ax.bar(x - 1.5*w, rmse, width=w, label="RMSE")
        ax.bar(x - 0.5*w, mae,  width=w, label="MAE")
        ax.bar(x + 0.5*w, mase, width=w, label="MASE")
        ax.bar(x + 1.5*w, r2,   width=w, label="R²")
        ax.set_xticks(x); ax.set_xticklabels(kinds)
        ax.set_title(f"드레인 {drain_id} — 모델 비교 지표")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(out), format="png")
        plt.close(fig)
        return out

    # ----- (옵션) 롤링 예측 & 임계 도달 시점 -----
    def _rollout_and_threshold(self, model, latest_block_feats, scaler, threshold_mm, horizon):
        """
        latest_block_feats: (seq_len, F) — 마지막 유효 시퀀스(리셋/NaN 없이)
        반환: dict { "forecast_mm": [..], "reach_index": int|None }
        """
        seq = latest_block_feats.copy()
        preds_scaled = []
        for _ in range(horizon):
            x_in = seq[np.newaxis, :, :]
            y_next_scaled = model.predict(x_in, verbose=0)  # (1,1)
            preds_scaled.append(y_next_scaled.item())
            # 다음 입력 업데이트: height_scaled만 갱신, 나머지 피처(예: cleaning_flag)는 0 가정
            next_feat = seq[-1].copy()
            next_feat[0] = y_next_scaled.item()
            if len(next_feat) > 1:
                next_feat[1:] = 0.0
            seq = np.vstack([seq[1:], next_feat])

        forecast_mm = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten().tolist()

        reach_index = None
        for i, v in enumerate(forecast_mm):
            if v >= threshold_mm:
                reach_index = i
                break
        return {"forecast_mm": forecast_mm, "reach_index": reach_index}

    def _save_threshold_plot(self, drain_id, last_value_mm, forecast_mm, reach_index):
        out = self.MODEL_DIR / f"forecast_threshold_drain_{drain_id}.png"
        x = np.arange(len(forecast_mm))
        fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=120)
        ax.plot(x, forecast_mm, marker='o', linewidth=1.6, label="롤링 예측(mm)")
        ax.hlines(self.THRESHOLD_MM, xmin=0, xmax=len(x)-1, linestyles='dashed', linewidth=1.2, label=f"임계 {self.THRESHOLD_MM}mm")
        ax.plot(0, last_value_mm, marker='s', label="현재", linestyle='None')

        if reach_index is not None:
            ax.vlines(reach_index, 0, max(self.THRESHOLD_MM, max(forecast_mm)), linestyles='dotted', linewidth=1.2, label="도달 시점")
            ax.text(reach_index, self.THRESHOLD_MM, f" +{reach_index}h ", va='bottom', ha='center')

        ax.set_title(f"드레인 {drain_id} — 임계 도달 롤링 예측")
        ax.set_xlabel("미래 시간(h)")
        ax.set_ylabel("예측 높이 (mm)")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", framealpha=0.9)
        fig.tight_layout(); fig.savefig(str(out), format="png"); plt.close(fig)
        return out

    # ===== 메인 =====
    def post(self, request):
        # 프리플라이트
        preflight = {"model_dir": str(self.MODEL_DIR), "ok": True, "error": None}
        try:
            self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            t = self.MODEL_DIR / "._w.tmp"; t.write_text("ok"); t.unlink(missing_ok=True)
        except Exception as e:
            preflight["ok"] = False; preflight["error"] = repr(e)

        drains = DrainInfo.objects.all()
        results = []

        for drain in drains:
            try:
                # ---- 초음파 로그 ----
                qs = SensorLog.objects.filter(drain=drain, sensor_type=self.SENSOR_HEIGHT).order_by('timestamp')
                if qs.count() < 50:
                    results.append({"drain": drain.id, "status": "데이터 부족"}); continue

                df = pd.DataFrame.from_records(qs.values('timestamp', 'value'))
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

                # 값 범위 (0~30). 0은 리셋
                df = df[(df['value'] >= 0) & (df['value'] <= 30)]
                # 1H 중앙값 리샘플
                df = df.resample('1H').median()

                # 세그 내부 1칸 결측만 ffill, 리셋은 0 유지
                v = df['value'].copy()
                is_reset = (v == 0)
                seg_id = is_reset.cumsum()
                pos_series = v.mask(is_reset)
                pos_filled = pos_series.groupby(seg_id).transform(lambda s: s.ffill(limit=1))
                df['value'] = pos_filled.where(~is_reset, 0.0)

                if len(df) < self.SEQ_LEN + 30:
                    results.append({"drain": drain.id, "status": "유효 샘플 부족"}); continue

                # ----- 피처 & 스케일러 -----
                feat_mat, values_raw, scaler = self._build_features(df)

                # ----- 시퀀스(사이클 내부) -----
                X, y_next_scaled = self._create_sequences_cycle(feat_mat, values_raw, seq_len=self.SEQ_LEN)
                if len(X) < 30:
                    results.append({"drain": drain.id, "status": "시퀀스 부족(사이클 내)"}); continue

                # 시간 순서 split
                n = len(X)
                i_tr = int(n * 0.70)
                i_va = int(n * 0.85)
                X_tr, y_tr = X[:i_tr], y_next_scaled[:i_tr]
                X_va, y_va = X[i_tr:i_va], y_next_scaled[i_tr:i_va]
                X_te, y_te = X[i_va:],    y_next_scaled[i_va:]

                if len(X_va) < 10 or len(X_te) < 10:
                    results.append({"drain": drain.id, "status": "검증/테스트 부족"}); continue

                # ----- 멀티 모델 학습/평가 -----
                metrics_map = {}
                model_objs = {}
                history_map = {}

                for kind in self.MODEL_KINDS:
                    model = self._build_model(kind, input_shape=(X.shape[1], X.shape[2]))
                    es = EarlyStopping(monitor='val_loss', patience=self.PATIENCE_ES, restore_best_weights=True)
                    rl = ReduceLROnPlateau(monitor='val_loss', patience=self.PATIENCE_RLR, factor=0.5,
                                           min_lr=self.MIN_LR, verbose=0)

                    h = model.fit(
                        X_tr, y_tr,
                        validation_data=(X_va, y_va),
                        epochs=self.EPOCHS, batch_size=self.BATCH_SIZE,
                        callbacks=[es, rl], verbose=0
                    )

                    # 검증 지표(선정용)
                    y_va_pred_scaled = model.predict(X_va, verbose=0)
                    m_va = self._evaluate_metrics(scaler, y_tr, y_va, y_va_pred_scaled)
                    metrics_map[kind] = {
                        "val_mae": m_va["mae"], "val_rmse": m_va["rmse"], "val_mase": m_va["mase"], "val_r2": m_va["r2"]
                    }
                    model_objs[kind] = model
                    history_map[kind] = {"loss": list(h.history.get("loss", [])),
                                         "val_loss": list(h.history.get("val_loss", []))}

                # 베스트 모델 선정 (val MASE → val RMSE)
                def sort_key(k):
                    return (metrics_map[k]["val_mase"], metrics_map[k]["val_rmse"])
                best_kind = sorted(self.MODEL_KINDS, key=sort_key)[0]
                best_model = model_objs[best_kind]

                # ----- 테스트 평가 (베스트 모델) -----
                y_te_pred_scaled = best_model.predict(X_te, verbose=0)
                m_te = self._evaluate_metrics(scaler, y_tr, y_te, y_te_pred_scaled)
                y_true = m_te["y_true"]; y_pred = m_te["y_pred"]

                # 저장 정책(나이브 대비)
                naive_pred = scaler.inverse_transform(X_te[:, -1, 0:1]).flatten()
                naive_rmse = float(np.sqrt(mean_squared_error(y_true, naive_pred)))
                naive_mae  = float(mean_absolute_error(y_true, naive_pred))
                beats = (m_te["mase"] <= 1.0 * self.BEAT_RATIO) or \
                        ((m_te["rmse"] <= naive_rmse * self.BEAT_RATIO) and (m_te["mae"] <= naive_mae * self.BEAT_RATIO))

                model_path  = self.MODEL_DIR / f"model_drain_{drain.id}.h5"
                scaler_path = self.MODEL_DIR / f"scaler_drain_{drain.id}.pkl"
                status_msg = "saved" if ((not self.SAVE_ONLY_IF_BETTER) or beats) else "skipped (baseline better/close)"
                if status_msg == "saved":
                    if model_path.exists(): model_path.unlink()
                    if scaler_path.exists(): scaler_path.unlink()
                    best_model.save(str(model_path)); joblib.dump(scaler, str(scaler_path))

                # ----- 성능 대시보드 지표 구성 -----
                # 테스트 기준 대시보드(권장): 베스트는 테스트 지표, 나머지는 검증 지표 사용
                metrics_for_dashboard = {}
                for k in self.MODEL_KINDS:
                    if k == best_kind:
                        metrics_for_dashboard[k] = {
                            "rmse": m_te["rmse"], "mae": m_te["mae"], "mase": m_te["mase"], "r2": m_te["r2"],
                            "acc2": m_te.get("acc_within_2mm", 0.0)
                        }
                    else:
                        metrics_for_dashboard[k] = {
                            "rmse": metrics_map[k]["val_rmse"], "mae": metrics_map[k]["val_mae"],
                            "mase": metrics_map[k]["val_mase"], "r2": metrics_map[k]["val_r2"],
                        }

                # ----- 산출물 저장 -----
                test_plot_path = None
                comp_plot_path = None
                forecast_plot_path = None
                metrics_dashboard_path = None

                try:
                    metrics_dashboard_path = str(self._save_metrics_dashboard(
                        drain.id, metrics_for_dashboard, best_kind
                    ))
                except Exception:
                    import logging; logging.exception("metrics dashboard save failed")

                if not self.SHOW_PERFORMANCE_ONLY:
                    if self.GENERATE_TEST_TIMELINE:
                        try:
                            test_plot_path = str(self._save_test_plot(
                                drain.id, y_true, y_pred,
                                m_te["rmse"], m_te["mae"], m_te["mase"]
                            ))
                        except Exception:
                            import logging; logging.exception("test plot save failed")

                    if self.GENERATE_MODEL_COMPARISON:
                        try:
                            comp_plot_path = str(self._save_model_comparison_plot(
                                drain.id,
                                {k: {"rmse": metrics_for_dashboard[k]["rmse"],
                                     "mae": metrics_for_dashboard[k]["mae"],
                                     "mase": metrics_for_dashboard[k]["mase"],
                                     "r2": metrics_for_dashboard[k]["r2"]} for k in self.MODEL_KINDS}
                            ))
                        except Exception:
                            import logging; logging.exception("comparison plot save failed")

                    if self.GENERATE_THRESHOLD_FORECAST:
                        try:
                            last_block = feat_mat[-self.SEQ_LEN:, :]  # (seq_len,F)
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
                        except Exception:
                            import logging; logging.exception("threshold forecast failed")

                # ----- summary JSON 저장 -----
                try:
                    summary = {
                        "drain": drain.id,
                        "best_model": best_kind,
                        "status": status_msg,
                        "val_metrics_per_model": metrics_map,
                        "test_metrics": {k: m_te[k] for k in ["mae","rmse","mape","smape","r2","mase","acc_within_2mm"]},
                        "plots": {
                            "metrics_dashboard": metrics_dashboard_path,
                            "test_plot": test_plot_path,
                            "model_comparison": comp_plot_path,
                            "threshold_forecast": forecast_plot_path
                        },
                        "params": {
                            "seq_len": self.SEQ_LEN,
                            "threshold_mm": self.THRESHOLD_MM,
                            "forecast_horizon": self.FORECAST_HORIZON
                        }
                    }
                    summ_path = self.MODEL_DIR / f"summary_drain_{drain.id}.json"
                    with open(summ_path, "w", encoding="utf-8") as f:
                        json.dump(summary, f, ensure_ascii=False, indent=2)
                except Exception:
                    import logging; logging.exception("summary json save failed")

                # ----- 응답 결과 -----
                results.append({
                    "drain": drain.id,
                    "status": status_msg,
                    "best_model": best_kind,
                    "beats_naive": bool(beats),
                    "seq_len": self.SEQ_LEN,
                    "count_train": int(len(X_tr)),
                    "count_val": int(len(X_va)),
                    "count_test": int(len(X_te)),
                    "val_metrics": metrics_map.get(best_kind, {}),
                    "test_metrics": {k: m_te[k] for k in ["mae","rmse","mape","smape","r2","mase","acc_within_2mm"]},
                    "plots": {
                        "metrics_dashboard": metrics_dashboard_path,
                        # 성능 지표만 원하면 아래 세 개는 None로 둡니다.
                        "test_plot": test_plot_path,
                        "model_comparison": comp_plot_path,
                        "threshold_forecast": forecast_plot_path
                    },
                })

            except Exception as e:
                import logging; logging.exception(f"Unexpected error for drain {drain.id}")
                results.append({"drain": drain.id, "status": f"error: {e!r}"})

        return Response({
            "preflight": preflight,
            "model_dir": str(self.MODEL_DIR.resolve()),
            "processed": len(drains),
            "results": results,
            "timestamp": timezone.now().isoformat(),
        }, status=200)