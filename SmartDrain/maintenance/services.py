# maintenance/services.py
import numpy as np
from datetime import timedelta
from django.utils import timezone
from django.conf import settings
from django.db import models

from .models import CleaningTask
from .prediction import forecast_next_hours
from accountapp.models import SensorLog

# ---- 설정값 ----
T_CLEAN     = getattr(settings, "DRAIN_THRESHOLD_VALUE", 5.0)
MODE        = getattr(settings, "DRAIN_THRESHOLD_MODE", "A")      # 'A' or 'B'
LEAD_HOURS  = getattr(settings, "DRAIN_LEAD_HOURS", 6)
# ★ due 이후 여유 버퍼(시간) - “다음 날 오전 가능” 정책
BUFFER_AFTER_DUE = getattr(settings, "DRAIN_WINDOW_END_BUFFER_HOURS", 12)

# services.py
from datetime import timedelta
from django.utils import timezone

# 기존 상수/설정 유지
# T_CLEAN, MODE, LEAD_HOURS 등은 그대로 사용한다고 가정

def find_due_time(preds):
    """
    ✅ 다음 시점의 1-step 예측값만으로 임계 판단.
    MODE:
      - 'A': 값이 작아질수록 나쁨  → next <= T_CLEAN 이면 due
      - 'B': 값이 커질수록 나쁨  → next >= T_CLEAN 이면 due
    """
    if not preds:
        return None
    nxt = preds[0]

    if str(MODE).upper() == 'A':
        breach = (nxt <= T_CLEAN)
    else:  # 'B' or 기타는 값↑ 나쁨으로 취급
        breach = (nxt >= T_CLEAN)

    if breach:
        # 바로 처리하고 싶다면 몇 분 뒤로 잡아둠(운영 정책에 맞게 조정)
        return timezone.now() + timedelta(minutes=5)
    return None


def compute_risk_score(preds, due_dt, now_dt):
    """ 남은시간↓, 최근 변화율↑ → 위험↑ """
    if not preds or not due_dt:
        return 0.0
    vals = np.array([v for (_, v) in preds], dtype=float)
    remain_h = max(0.1, (due_dt - now_dt).total_seconds()/3600.0)
    slope = float(np.mean(np.diff(vals[-6:]))) if len(vals) >= 7 else 0.0
    base = 100.0 / remain_h
    return base + 10.0 * abs(slope)

def generate_task_for_drain(drain):
    """
    drain 하나에 대해 (1) 현재값 즉시 임계 → (2) 24h 예측 임계 검사 → 티켓 생성
    window_end 는 due + BUFFER_AFTER_DUE 로 설정 (다음 날 오전 허용)
    """
    now = timezone.now()

    # --- 1) 현재 관측값 즉시 감지 ---
    last_log = SensorLog.objects.filter(
        drain=drain, sensor_type='초음파 센서'
    ).order_by('-timestamp').first()

    if last_log is not None:
        v = float(last_log.value)
        current_breach = (MODE.upper() == 'A' and v <= T_CLEAN) or (MODE.upper() == 'B' and v >= T_CLEAN)
        if current_breach:
            due = now + timedelta(minutes=5)  # 즉시 처리
            risk = 100.0
            window_start = now
            window_end   = due + timedelta(hours=BUFFER_AFTER_DUE)  # ★ 변경점
            return CleaningTask.objects.create(
                drain=drain,
                lat=float(drain.latitude),         # ★ latitude / longitude 사용
                lng=float(drain.longitude),
                predicted_due=due,
                estimated_duration_min=20,
                risk_score=risk,
                window_start=window_start,
                window_end=window_end,
                status="pending",
            )

    # --- 2) 예측 기반 ---
    preds = forecast_next_hours(drain)
    if not preds:
        return None

    due = find_due_time(preds)
    if not due:
        return None

    risk = compute_risk_score(preds, due, now)
    window_start = due - timedelta(hours=LEAD_HOURS)
    window_end   = due + timedelta(hours=BUFFER_AFTER_DUE)  # ★ 변경점

    return CleaningTask.objects.create(
        drain=drain,
        lat=float(drain.latitude),             # ★ latitude / longitude 사용
        lng=float(drain.longitude),
        predicted_due=due,
        estimated_duration_min=20,
        risk_score=risk,
        window_start=window_start,
        window_end=window_end,
        status="pending",
    )

def generate_tasks_for_all_drains(DrainInfoModel):
    created = 0
    skipped = 0
    for drain in DrainInfoModel.objects.all():
        t = generate_task_for_drain(drain)
        if t:
            created += 1
        else:
            skipped += 1
    return {"created": created, "skipped": skipped}
