#views.py
from .serializers import CleaningTaskSerializer
from .services import generate_tasks_for_all_drains
from .scheduler import greedy_schedule_for_date
from accountapp.models import DrainInfo  # ★ 교체
from rest_framework.views import APIView
from rest_framework.generics import ListAPIView, RetrieveAPIView
from rest_framework.permissions import AllowAny
from .models import Crew
from .serializers import CrewSerializer

class CrewListView(ListAPIView):
    queryset = Crew.objects.all().order_by("id")
    serializer_class = CrewSerializer
    permission_classes = [AllowAny]

class CrewDetailView(RetrieveAPIView):
    queryset = Crew.objects.all()
    serializer_class = CrewSerializer
    permission_classes = [AllowAny]

def _build_ct_schedule_date_filter(target_date):
    """
    CleaningTask에서 스케줄 시점을 나타내는 필드 자동 감지:
    1) scheduled_date (DateField)
    2) scheduled_start (DateTimeField)
    3) start (DateTimeField)
    위 우선순위로 해당 날짜 필터 dict를 반환. 없으면 None.
    """
    Task = CleaningTask
    fields = {f.name: f for f in Task._meta.get_fields() if hasattr(f, "name")}
    sod = timezone.make_aware(datetime.combine(target_date, time.min))
    eod = sod + timedelta(days=1)

    if "scheduled_date" in fields and isinstance(fields["scheduled_date"], DateField):
        return {"scheduled_date": target_date}

    if "scheduled_start" in fields and isinstance(fields["scheduled_start"], DateTimeField):
        return {"scheduled_start__gte": sod, "scheduled_start__lt": eod}

    if "start" in fields and isinstance(fields["start"], DateTimeField):
        return {"start__gte": sod, "start__lt": eod}

    return None


class PredictAndGenerateTasksView(APIView):
    """
    모든 배수구에 대해 예측 → 임계 도달 시점 계산 → CleaningTask 생성 → (바로) 당일 스케줄 배정
    - 요청 파라미터:
        date: "YYYY-MM-DD" (없으면 오늘)
        overwrite: true/false (기본 False)
        include_forecast: true/false (기본 True)  # (지금은 미사용)
        horizon_hours: int (기본 72)               # (지금은 미사용)
    """

    def post(self, request):
        """
        모든 배수구에 대해 예측 → 임계 도달 시점 계산 → CleaningTask 생성 → (바로) 당일 스케줄 배정
        - Zero-Guard(리셋=0 처리) 백스톱 포함
        - 🔧 수정점:
            (A) 마지막 원시 값(raw)이 임계 이상이면 즉시 티켓 발급 (현재값 우선 가드)
            (B) 시간 리샘플 df_h는 마지막 버킷을 원시 마지막 값으로 덮어써 현재값이 0으로 깎이지 않게 함
        - DB 스키마 변경 없음(기존 필드만 사용)
        """
        import numpy as np
        import pandas as pd
        from datetime import datetime, timedelta
        from django.db import transaction
        from django.utils import timezone
        from rest_framework import status
        from rest_framework.response import Response

        # ----- 파라미터 -----
        date_str = request.data.get("date")
        overwrite = bool(request.data.get("overwrite"))
        include_forecast = request.data.get("include_forecast", True)
        include_forecast = True if include_forecast is None else bool(include_forecast)
        horizon_hours = int(request.data.get("horizon_hours", 72))

        # 날짜 파싱 (미지정 시 오늘)
        if date_str:
            try:
                target_date = datetime.fromisoformat(str(date_str).strip()).date()
            except Exception:
                return Response({"detail": "date 형식은 YYYY-MM-DD"}, status=400)
        else:
            target_date = timezone.localdate()

        body = {"date": str(target_date)}

        # =========================
        # Zero-Guard 유틸 (내부 함수)
        # =========================
        def _extract_post_reset_series(df: pd.DataFrame, value_col="value", min_run=3):
            """
            df: index=datetime(1h), column 'value'
            - 마지막 리셋(==0) 이후 구간만 사용
            - 고립된 0(앞뒤가 >0인데 가운데만 0)은 NaN 처리
            - 유효 구간 길이가 min_run 미만이면 빈 Series 반환
            """
            if value_col not in df.columns or df.empty:
                return pd.Series(dtype=float, name=value_col), None

            s = df[value_col].astype(float).copy()

            # (1) 고립된 0 제거
            s_shift_f = s.shift(-1)
            s_shift_b = s.shift(1)
            isolated_zero = (s == 0) & (s_shift_f > 0) & (s_shift_b > 0)
            s.loc[isolated_zero] = np.nan

            # (2) 마지막 리셋(=0) 인덱스
            reset_points = s.index[s == 0]
            last_reset_at = reset_points.max() if len(reset_points) else None

            # (3) 마지막 리셋 이후만 사용
            if last_reset_at is not None:
                s = s.loc[s.index > last_reset_at]

            # (4) 짧은 결측 보강(한 칸만)
            s = s.ffill(limit=1)

            if s.dropna().shape[0] < min_run:
                return pd.Series(dtype=float, name=value_col), last_reset_at

            s.name = value_col
            return s, last_reset_at

        def _should_issue_ticket_now(df_hourly: pd.DataFrame,
                                     latest_raw_value: float = None,
                                     threshold_mm: float = 25.0,
                                     hours_ahead: int = 24,
                                     safety_margin_mm: float = 0.0) -> bool:
            """
            임계 도달 여부 판단(Zero-Guard 적용):
            - (우선) latest_raw_value가 임계-마진 이상이면 즉시 True  ← 🔴 추가
            - 마지막 리셋 이후 구간만 사용
            - 현재 값이 임계-마진 이상이면 True
            - 아니면 최근 기울기 기반 선형외삽으로 hours_ahead 내 임계 도달 예측 시 True
            """
            # 🔴 현재 원시값 우선 가드
            if latest_raw_value is not None and latest_raw_value >= (threshold_mm - safety_margin_mm):
                return True

            s, _ = _extract_post_reset_series(df_hourly, "value", min_run=3)
            if s.empty:
                return False

            cur = float(s.dropna().iloc[-1])
            if cur >= (threshold_mm - safety_margin_mm):
                return True

            # 최근 k시간 추세로 외삽
            k = 6
            recent = s.dropna().tail(k)
            if recent.shape[0] >= 2:
                y = recent.values
                x = np.arange(len(recent), dtype=float)
                A = np.vstack([x, np.ones_like(x)]).T
                m, b = np.linalg.lstsq(A, y, rcond=None)[0]  # y = m x + b
                y_fore = m * (len(recent) - 1 + hours_ahead) + b
                if y_fore >= threshold_mm:
                    return True

            return False

        # ===== 1) 예측 → 임계 도달 계산 → CleaningTask 생성 =====
        # '이번 요청에서 새로 만든 작업'만 잡기 위한 스냅샷
        before_ids = set(CleaningTask.objects.values_list("id", flat=True))
        try:
            with transaction.atomic():
                gen_result = generate_tasks_for_all_drains(DrainInfo)
        except Exception as e:
            return Response({"detail": f"CleaningTask 생성 실패: {e!r}"}, status=500)

        after_ids = set(CleaningTask.objects.values_list("id", flat=True))
        created_ids = sorted(list(after_ids - before_ids))  # ← 이번 호출에서 막 생성된 ID만

        body["generate"] = {
            **(gen_result if isinstance(gen_result, dict) else {"summary": gen_result}),
            "created_ids": created_ids,
            "created_count_detected": len(created_ids),
        }

        # ===== 1.1) Zero-Guard 백스톱: 예측/임계 단계에서 누락된 티켓 보강 생성 =====
        drain_ids_all = set(DrainInfo.objects.values_list("id", flat=True))
        task_rows = CleaningTask.objects.filter(id__in=created_ids).values("id", "drain_id")
        drain_ids_already_issued = set(r["drain_id"] for r in task_rows)
        drain_ids_missing = sorted(list(drain_ids_all - drain_ids_already_issued))

        # 최근 N일 데이터로 오늘 임계 도달 가능성 보수 판정
        lookback_days = 7
        start_dt = timezone.make_aware(
            datetime.combine(target_date - timedelta(days=lookback_days), datetime.min.time()))
        end_dt = timezone.make_aware(datetime.combine(target_date, datetime.max.time()))

        to_force_issue = []
        for did in drain_ids_missing:
            qs = (SensorLog.objects
                  .filter(drain_id=did, sensor_type='초음파 센서',
                          timestamp__gte=start_dt, timestamp__lte=end_dt)
                  .order_by('timestamp')
                  .values('timestamp', 'value'))
            if qs.count() < 1:
                continue

            df = pd.DataFrame.from_records(qs)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            # 유효값 범위
            df = df[(df['value'] >= 0) & (df['value'] <= 30)]
            if df.empty:
                continue

            # 🔴 현재 원시 마지막 값 확보 (예: 27이면 바로 True로 갈 수 있게)
            latest_raw_value = float(df['value'].iloc[-1])

            # 1h 리샘플 (중앙값) + 🔴 마지막 버킷은 원시 마지막 값으로 덮어쓰기
            df_h = df.resample('1h').median()
            last_hour = df.index[-1].floor('h')
            # 마지막 버킷이 존재하면 원시 마지막 값으로 강제 반영 (median에 0 섞여도 현재값 보존)
            if last_hour in df_h.index:
                df_h.at[last_hour, 'value'] = latest_raw_value
            else:
                # resample 결과에 마지막 시간 인덱스가 없다면 행 추가
                df_h.loc[last_hour, 'value'] = latest_raw_value
                df_h = df_h.sort_index()

            # 🔒 제로-가드 판정: (A) 현재 원시값, (B) post-reset 추세
            if _should_issue_ticket_now(
                    df_hourly=df_h,
                    latest_raw_value=latest_raw_value,  # ← 🔴 추가
                    threshold_mm=25.0,
                    hours_ahead=24,
                    safety_margin_mm=0.0
            ):
                to_force_issue.append(did)

        forced_task_ids = []
        if to_force_issue:
            try:
                with transaction.atomic():
                    for did in to_force_issue:
                        # 당일 동일 하수구 스케줄/티켓 있으면 패스(중복 방지)
                        date_filter = _build_ct_schedule_date_filter(target_date)
                        exists = CleaningTask.objects.filter(drain_id=did, **(date_filter or {})).exists()
                        if exists and not overwrite:
                            continue

                        # ✅ 스키마 변경 없이 생성(기존 필드만)
                        ct = CleaningTask.objects.create(
                            drain_id=did,
                            title="[AUTO] 임계 도달 예상 (Zero-Guard backstop)",
                            status="NEW",
                            due_date=target_date,  # 기존 로직이 due_date 기준이면 그대로 사용
                        )
                        forced_task_ids.append(ct.id)
            except Exception as e:
                body.setdefault("warnings", []).append(f"Zero-Guard 보강 생성 실패: {e!r}")

        # 생성 리스트에 보강 생성분 합치기
        if forced_task_ids:
            created_ids = sorted(list(set(created_ids) | set(forced_task_ids)))

        # ===== 1.5) 스케줄 중복 방지 (같은 날짜·같은 하수구 중복 배정 금지) =====
        # 새로 만든 task들의 drain 매핑
        task_rows = CleaningTask.objects.filter(id__in=created_ids).values("id", "drain_id")
        task_to_drain = {r["id"]: r["drain_id"] for r in task_rows}
        drains_in_created = set(task_to_drain.values())

        # 기존 헬퍼로 날짜 필터 구성
        date_filter = _build_ct_schedule_date_filter(target_date)

        existing_qs = CleaningTask.objects.filter(drain_id__in=drains_in_created)
        if date_filter:
            existing_qs = existing_qs.filter(**date_filter)
        # (필요 시 상태 제외)
        # existing_qs = existing_qs.exclude(status__in=["DONE", "CANCELLED"])

        already_scheduled_drains = set(
            existing_qs.values_list("drain_id", flat=True).distinct()
        )

        skipped_task_ids = []
        filtered_ids = list(created_ids)

        if not overwrite:
            # 이미 당일에 스케줄된 하수구의 신규 task는 스킵
            filtered_ids = [
                tid for tid in created_ids
                if task_to_drain.get(tid) not in already_scheduled_drains
            ]
            skipped_task_ids = [tid for tid in created_ids if tid not in filtered_ids]
        else:
            # 덮어쓰기 모드면: 같은 하수구의 당일 기존 CleaningTask를 삭제 후 진행
            drains_to_clear = {task_to_drain[tid] for tid in created_ids}
            with transaction.atomic():
                del_qs = CleaningTask.objects.filter(drain_id__in=drains_to_clear)
                if date_filter:
                    del_qs = del_qs.filter(**date_filter)
                # 방금 생성된 것 제외
                del_qs = del_qs.exclude(id__in=created_ids)
                del_qs.delete()
            filtered_ids = list(created_ids)

        body["schedule_guard"] = {
            "overwrite": overwrite,
            "already_scheduled_drains": list(already_scheduled_drains),
            "skipped_task_ids": skipped_task_ids,
            "scheduled_task_candidates": filtered_ids,
            "zero_guard_forced_task_ids": forced_task_ids,
            "zero_guard_forced_count": len(forced_task_ids),
        }

        if not filtered_ids:
            # 이번 호출에서 실제로 스케줄링할 대상이 없으면 조기 반환
            body["schedules"] = {}
            body["overwrite"] = overwrite
            return Response(body, status=status.HTTP_200_OK)

        # ===== 2) (바로) 당일 스케줄 배정 — 중복 필터링 통과분만 =====
        try:
            schedules = greedy_schedule_for_date(
                target_date=target_date,
                overwrite=overwrite,
                task_ids=filtered_ids,  # ★ 중복 제거된 목록만
                assign_all=False,
                allow_overtime=False,
                force_assign_remaining=True,
                # min_gap_min=0,
                # lunch_break=("12:00","13:00"),
            )
        except Exception as e:
            return Response(
                {
                    "detail": f"스케줄 배정 실패: {e!r}",
                    "generate": body.get("generate"),
                    "schedule_guard": body.get("schedule_guard"),
                },
                status=500
            )

        # 응답 변환
        result = {}
        for crew_id, events in schedules.items():
            result[str(crew_id)] = [
                {"task_id": tid, "start": st.isoformat(), "end": et.isoformat()}
                for (tid, st, et) in events
            ]
        body["schedules"] = result
        body["overwrite"] = overwrite

        # ===== 3) (옵션) include_forecast/horizon_hours 사용 지점 (현재 미사용) =====

        return Response(body, status=status.HTTP_200_OK)


class TaskListView(APIView):
    """
    작업 목록 조회 (status 필터 가능: pending/scheduled/done)
    """
    def get(self, request):
        status_q = request.query_params.get("status")
        qs = CleaningTask.objects.all().order_by("-created_at")
        if status_q:
            qs = qs.filter(status=status_q)
        ser = CleaningTaskSerializer(qs, many=True)
        return Response(ser.data, status=200)

class DaySchedulePreviewView(APIView):
    """
    조회 전용: 산출 결과를 바로 계산해 미리보기
    GET /maintenance/schedules/day/?date=YYYY-MM-DD
    """
    def get(self, request):
        date_str = request.query_params.get("date")
        if date_str:
            try:
                target_date = datetime.fromisoformat(date_str).date()
            except Exception:
                return Response({"detail": "date 형식은 YYYY-MM-DD"}, status=400)
        else:
            target_date = timezone.localdate()

        schedules = greedy_schedule_for_date(target_date)
        result = {}
        for crew_id, events in schedules.items():
            result[str(crew_id)] = [
                {"task_id": tid, "start": st.isoformat(), "end": et.isoformat()}
                for (tid, st, et) in events
            ]
        return Response({"date": str(target_date), "preview": result}, status=200)

class PredictDebugView(APIView):
    """
    드레인별 예측/임계 체크 디버그:
    - 모델/스케일러 존재 여부
    - 최근 유효 샘플 길이
    - 마지막 관측값(last_observed)
    - 예측 개수/샘플
    - 임계/모드/결정된 due
    - 최종 create 혹은 skip 이유
    """
    def get(self, request):
        from django.conf import settings
        from django.utils import timezone
        from datetime import timedelta
        from accountapp.models import DrainInfo, SensorLog
        from .prediction import load_model_and_scaler, prepare_latest_sequence, forecast_next_hours
        from .services import find_due_time, T_CLEAN, MODE, LEAD_HOURS
        import os

        results = []
        for drain in DrainInfo.objects.all():
            info = {"drain_id": drain.id}
            # 모델/스케일러 파일 존재
            model_path  = os.path.join("trained_models", f"model_drain_{drain.id}.h5")
            scaler_path = os.path.join("trained_models", f"scaler_drain_{drain.id}.pkl")
            info["has_model"] = os.path.exists(model_path)
            info["has_scaler"] = os.path.exists(scaler_path)

            # 최근 시퀀스 상태
            seq, last_idx = prepare_latest_sequence(drain, getattr(settings, "MODEL_SEQ_LEN", 24))
            info["recent_valid_len"] = 0 if seq is None else len(seq)

            # 마지막 관측값(리샘플 전 원시 마지막 한 개)
            last_log = SensorLog.objects.filter(drain=drain, sensor_type='초음파 센서').order_by('-timestamp').first()
            info["last_observed_ts"] = last_log.timestamp.isoformat() if last_log else None
            info["last_observed_value"] = float(last_log.value) if last_log else None

            # 예측
            preds = forecast_next_hours(drain)
            info["pred_count"] = len(preds)
            info["pred_sample"] = preds[:3]  # 앞 3개만 샘플

            # 임계/모드
            info["mode"] = MODE
            info["threshold"] = T_CLEAN

            # 즉시 임계(현재값) 감지
            now_due = None
            if info["last_observed_value"] is not None:
                if (MODE.upper() == 'A' and info["last_observed_value"] <= T_CLEAN) or \
                   (MODE.upper() == 'B' and info["last_observed_value"] >= T_CLEAN):
                    now_due = timezone.now() + timedelta(minutes=5)  # 즉시 처리로 간주
            info["current_breach"] = bool(now_due is not None)

            # 예측 기반 due
            due_pred = find_due_time(preds) if preds else None
            info["due_from_pred"] = due_pred.isoformat() if due_pred else None

            # 최종 판단(현재값 우선 → 예측)
            final_due = now_due or due_pred
            info["final_due"] = final_due.isoformat() if final_due else None
            if not final_due:
                info["decision"] = "skip: no due in current or next 24h"
            else:
                info["decision"] = "create: due determined"

            results.append(info)

        return Response(results, status=200)

class DayScheduleDetailedView(APIView):
    """
    상세 스케줄: 작업자/하수구/시간/좌표 전부 함께 반환
    GET /maintenance/schedules/day/detailed/?date=YYYY-MM-DD
    """
    def get(self, request):
        date_str = request.query_params.get("date")
        if date_str:
            try:
                target_date = datetime.fromisoformat(date_str).date()
            except Exception:
                return Response({"detail":"date 형식은 YYYY-MM-DD"}, status=400)
        else:
            target_date = timezone.localdate()

        # 그 날짜에 scheduled 된 작업만 조회
        day_start = timezone.make_aware(datetime.combine(target_date, datetime.min.time()))
        day_end   = timezone.make_aware(datetime.combine(target_date, datetime.max.time()))

        tasks = (CleaningTask.objects
                 .select_related("assigned_crew","drain")
                 .filter(status="scheduled", scheduled_start__gte=day_start, scheduled_start__lte=day_end)
                 .order_by("assigned_crew_id","scheduled_start","id"))

        # crew별 그룹핑
        result = {}
        for t in tasks:
            crew = t.assigned_crew
            if crew is None:
                crew_key = "unassigned"
            else:
                crew_key = str(crew.id)
                if crew_key not in result:
                    result[crew_key] = {
                        "crew": {
                            "id": crew.id,
                            "name": crew.name,
                            "home": {"lat": crew.home_lat, "lng": crew.home_lng},
                            "shift": {"start": str(crew.shift_start), "end": str(crew.shift_end)},
                        },
                        "jobs": []
                    }
            # drain 상세
            d = t.drain
            job = {
                "task_id": t.id,
                "scheduled": {"start": t.scheduled_start.isoformat() if t.scheduled_start else None,
                              "end":   t.scheduled_end.isoformat()   if t.scheduled_end   else None},
                "window":    {"start": t.window_start.isoformat() if t.window_start else None,
                              "end":   t.window_end.isoformat()   if t.window_end   else None},
                "drain": {
                    "id": d.id,
                    "name": d.name,
                    "region": d.region,
                    "sub_region": d.sub_region,
                    "lat": d.latitude,
                    "lng": d.longitude,
                },
                "task_point": {"lat": t.lat, "lng": t.lng},   # 보통 drain 좌표와 동일
                "predicted_due": t.predicted_due.isoformat(),
                "risk_score": t.risk_score,
                "estimated_duration_min": t.estimated_duration_min,
            }
            # crew_key 없으면(배정 누락) 기본 버킷
            if t.assigned_crew is None:
                result.setdefault("unassigned", {"crew": None, "jobs":[]})["jobs"].append(job)
            else:
                result[crew_key]["jobs"].append(job)

        return Response({"date": str(target_date), "schedules": result}, status=200)



class DayRouteGeoJSONView(APIView):

        def get(self, request):
            # 1) 날짜 파라미터 파싱 (빈칸/줄바꿈 제거)
            date_str = (request.query_params.get("date") or "").strip()
            if date_str:
                try:
                    target_date = datetime.fromisoformat(date_str).date()
                except Exception:
                    return Response({"detail": "date 형식은 YYYY-MM-DD"}, status=400)
            else:
                target_date = timezone.localdate()

            # 2) 당일 시간 범위 생성 (타임존 포함)
            tz = timezone.get_current_timezone()
            day_start = timezone.make_aware(datetime.combine(target_date, datetime.min.time()), tz)
            day_end = timezone.make_aware(datetime.combine(target_date, datetime.max.time()), tz)

            features = []

            # 3) 모든 crew에 대해, 해당 날짜 스케줄(시간순) 가져와 경로 구성
            for crew in Crew.objects.all():
                # 이 crew에게 배정된, 해당 날짜의 스케줄된 작업들
                qs = (CleaningTask.objects
                      .select_related("drain")
                      .filter(assigned_crew=crew,
                              scheduled_start__gte=day_start,
                              scheduled_start__lte=day_end)
                      .order_by("scheduled_start", "id"))

                if not qs.exists():
                    # 이 crew는 그 날 배정이 없으면 Feature 생성 안 함
                    continue

                # 4) GeoJSON LineString 좌표 배열 (주의: [lng, lat] 순서)
                coordinates = [[crew.home_lng, crew.home_lat]]  # 출발지(크루 거점)
                stops = [{"type": "start", "name": f"{crew.name} 출발"}]

                for t in qs:
                    # 작업지 좌표 추가 (CleaningTask.lat/lng는 float로 저장됨)
                    coordinates.append([t.lng, t.lat])
                    stops.append({
                        "type": "job",
                        "task_id": t.id,
                        "drain_id": t.drain_id,
                        "drain_name": t.drain.name,
                        "scheduled_start": t.scheduled_start.isoformat() if t.scheduled_start else None,
                        "scheduled_end": t.scheduled_end.isoformat() if t.scheduled_end else None,
                    })

                # 5) crew 하나 = Feature 하나
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coordinates},
                    "properties": {
                        "crew_id": crew.id,
                        "crew_name": crew.name,
                        "stops": stops
                    }
                })

            # 6) FeatureCollection으로 묶어서 반환
            return Response({"type": "FeatureCollection", "features": features}, status=200)

        # maintenance/views.py (예시 목록 API)
from accountapp.models import SensorLog   # SensorLog가 있는 앱 경로
from datetime import datetime, date as date_cls, time, timedelta
from django.db.models import OuterRef, Subquery, DateTimeField, DateField
# views.py
from datetime import datetime, date as date_cls, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from django.apps import apps
from django.conf import settings
from django.db import transaction, connection
from django.db.models import OuterRef, Subquery
from django.utils import timezone

from rest_framework.response import Response
from rest_framework.views import APIView

# 필요 모델 import (프로젝트 구조에 맞게 경로 조정)
from .models import CleaningTask, Crew  # 예시: 같은 앱 내 모델일 경우
# 만약 다른 앱이라면: from core.models import CleaningTask, Crew  등으로 수정


# views.py

import math
from datetime import datetime, timedelta, date as date_cls
from decimal import Decimal
from zoneinfo import ZoneInfo

from django.conf import settings
from django.apps import apps
from django.db import connection, transaction
from django.db.models import OuterRef, Subquery
from django.utils import timezone

from rest_framework.response import Response
from rest_framework.views import APIView

# 필요에 맞게 모델 import 경로를 조정하세요.
# 예시:
# from .models import Crew, CleaningTask
from .models import Crew, CleaningTask  # ← 실제 앱으로 변경


class CrewTasksView(APIView):
    """
    GET  /api/crews/tasks?date=YYYY-MM-DD&include_done=false[&crew=팀A|all][&crew_id=1][&order=nearest|time|none]
         - crew/crew_id 생략 또는 crew=all 이면 모든 크루를 그룹핑해서 반환
         - order 파라미터:
           * nearest(기본): 크루 위치(lat/lng)에서 가까운 작업부터 정렬
           * time/none: scheduled_start 기준 정렬

    POST /api/crews/tasks?date=YYYY-MM-DD[&order=nearest|time|none]
         body: { "task_id": 222, "when": "2025-09-02T10:20:00+09:00"(옵션) }
         - crew 미지정이어도 해당 날짜 범위에서 task_id 를 찾아 done 처리
         - ✅ done 처리 시, 해당 하수구 센서 로그(SensorLogs)에 ("초음파 센서", 0.0) 1건 추가
           (센서 로그의 timestamp는 항상 '현재 KST'로 저장, 세션 타임존 +09:00 고정)
         - 응답 task 항목에만 risk_score 포함 (요약 X)
         - ✅ 응답 JSON에서는 각 작업 시작 시간을 최소 2시간(slot) 간격으로 보정(겹치지 않게 표시)
    """

    # ---------- helpers ----------

    def _parse_date(self, date_str):
        """
        허용 예시:
        - '2025-09-02'
        - '2025/09/02'
        - '2025-9-2'
        - '2025-09-02T10:20:00+09:00'
        - '2025-09-02T01:20:00Z'
        """
        if not date_str:
            return timezone.localdate()

        s = str(date_str).strip()
        if not s:
            return timezone.localdate()

        s = s.replace("/", "-")  # 슬래시 허용
        try:
            if "T" not in s and " " not in s and s.count("-") == 2 and len(s) <= 10:
                y, m, d = [int(p) for p in s.split("-")]
                return date_cls(y, m, d)
        except Exception:
            pass

        s_dt = s.replace("Z", "+00:00")  # ISO8601 + Z 허용
        try:
            return datetime.fromisoformat(s_dt).date()
        except Exception:
            return None

    def _bounds(self, target_date):
        tz = timezone.get_current_timezone()
        return (
            timezone.make_aware(datetime.combine(target_date, datetime.min.time()), tz),
            timezone.make_aware(datetime.combine(target_date, datetime.max.time()), tz),
        )

    def _resolve_crew(self, request):
        """
        반환: Crew 인스턴스 또는 None
        - crew=all 이면 None 반환 → 전체 처리 분기
        """
        crew_q = request.query_params.get("crew") or request.data.get("crew")
        if crew_q:
            if str(crew_q).lower() in ("all", "*"):
                return None
            try:
                cid = int(str(crew_q))
                c = Crew.objects.filter(id=cid).first()
                if c:
                    return c
            except Exception:
                pass
            c = Crew.objects.filter(name=crew_q).first()
            if c:
                return c

        crew_id_q = request.query_params.get("crew_id") or request.data.get("crew_id")
        if crew_id_q:
            try:
                return Crew.objects.get(id=int(crew_id_q))
            except Crew.DoesNotExist:
                return None

        return Crew.objects.first() if Crew.objects.count() == 1 else None

    def _iso_local(self, dt):
        """응답 JSON에 로컬(Asia/Seoul) ISO8601로 보여주기"""
        if not dt:
            return None
        if timezone.is_naive(dt):
            dt = timezone.make_aware(dt, timezone.get_current_timezone())
        return timezone.localtime(dt).isoformat()

    # ---------- distance helpers ----------

    def _haversine_km(self, lat1, lng1, lat2, lng2):
        """위도/경도(도)를 받아 대략적인 구면거리(km) 계산"""
        if None in (lat1, lng1, lat2, lng2):
            return float("inf")
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def _order_by_nearest(self, start_lat, start_lng, tasks):
        """
        단순 '최근접 이웃' 탐욕 알고리즘으로 tasks를 정렬.
        입력: 모델 인스턴스 리스트(이미 qs → list)
        반환: 정렬된 리스트
        """
        remain = tasks[:]
        ordered = []
        cur_lat, cur_lng = start_lat, start_lng

        while remain:
            best_i = None
            best_d = float("inf")
            for i, t in enumerate(remain):
                d = self._haversine_km(cur_lat, cur_lng, getattr(t, "lat", None), getattr(t, "lng", None))
                if d < best_d:
                    best_d = d
                    best_i = i
            nxt = remain.pop(best_i)
            ordered.append(nxt)
            cur_lat = getattr(nxt, "lat", cur_lat)
            cur_lng = getattr(nxt, "lng", cur_lng)
        return ordered

    # ---------- risk 주입 관련 helpers ----------

    def _get_draininfo_model(self):
        """
        DrainInfo 모델 탐색: settings.DRAININFO_MODEL='app_label.ModelName' 이 설정돼 있으면 우선 사용.
        없으면 흔한 후보들을 시도.
        """
        path = getattr(settings, "DRAININFO_MODEL", None)  # 예: "core.DrainInfo"
        if path:
            try:
                return apps.get_model(path)
            except LookupError:
                pass
        for cand in ["core.DrainInfo", "drains.DrainInfo", "drain.DrainInfo"]:
            try:
                return apps.get_model(cand)
            except LookupError:
                continue
        return None

    def _pick_risk_field(self, Model):
        """
        DrainInfo에서 위험도를 나타내는 필드명을 선택.
        우선순위: risk_score → risk → risk_level
        """
        names = {getattr(f, "name", None) for f in Model._meta.get_fields()}
        for n in ("risk_score", "risk", "risk_level"):
            if n in names:
                return n
        return None

    def _annotate_risk(self, qs):
        """
        QuerySet에 risk_score를 보장:
        - CleaningTask에 risk_score/risk/risk_level 중 하나가 있으면 그대로 사용(별도 annotate 불필요)
        - 없으면 DrainInfo에서 Subquery로 risk를 끌어와 'risk_score' 라는 이름으로 annotate
        """
        task_fields = {getattr(f, "name", None) for f in CleaningTask._meta.get_fields()}
        if any(n in task_fields for n in ("risk_score", "risk", "risk_level")):
            return qs  # Task 자체에 있으면 주입 불필요

        DrainInfo = self._get_draininfo_model()
        if not DrainInfo:
            return qs

        risk_field = self._pick_risk_field(DrainInfo)
        if not risk_field:
            return qs

        di_fields = {getattr(f, "name", None) for f in DrainInfo._meta.get_fields()}
        di_pk = "id" if "id" in di_fields else ("drain_id" if "drain_id" in di_fields else None)
        if not di_pk:
            return qs

        # CleaningTask에 drain_id가 있다고 가정(다르면 여기 매핑 수정)
        subq = DrainInfo.objects.filter(**{di_pk: OuterRef("drain_id")}).values(risk_field)[:1]
        return qs.annotate(risk_score=Subquery(subq))

    def _task_to_dict(self, t):
        """
        응답용 dict 변환 (task 항목에만 risk_score 포함).
        우선순위: t.risk_score → t.risk → t.risk_level (없으면 None)
        Decimal은 float로 변환.
        """
        risk_val = None
        for name in ("risk_score", "risk", "risk_level"):
            if hasattr(t, name):
                val = getattr(t, name)
                if val is not None:
                    if isinstance(val, Decimal):
                        try:
                            val = float(val)
                        except Exception:
                            pass
                    risk_val = val
                    break

        return {
            "id": t.id,
            "status": t.status,
            "start": self._iso_local(t.scheduled_start),
            "end": self._iso_local(t.scheduled_end),
            "drain_id": t.drain_id,
            "lat": t.lat,
            "lng": t.lng,
            "risk_score": risk_val,  # ← 각 task에만 존재
        }

    # ---------- display spread helper ----------

    def _spread_for_display(self, tasks, slot_hours=2, default_duration_minutes=None):
        """
        응답 JSON에서만 작업 시작/끝 시간을 '겹치지 않게' 보정합니다.
        - slot_hours: 시작 시간 간 최소 간격(기본 2시간)
        - default_duration_minutes: 작업 시간이 명시되지 않은 경우 사용할 기본 duration
          (None이면: t.scheduled_end - t.scheduled_start가 유효하면 그걸 쓰고, 없으면 slot_hours를 duration으로 간주)
        반환: [{"id":..., "start":..., "end":..., ...}, ...]  ← _task_to_dict 형식과 동일
        """
        tz = timezone.get_current_timezone()
        slot = timedelta(hours=slot_hours)

        adjusted = []
        next_free = None

        for t in tasks:
            orig_start = t.scheduled_start
            orig_end = t.scheduled_end

            # timezone aware 보장
            if timezone.is_naive(orig_start):
                orig_start = timezone.make_aware(orig_start, tz)

            # duration 계산
            if orig_end:
                if timezone.is_naive(orig_end):
                    orig_end = timezone.make_aware(orig_end, tz)
                dur = orig_end - orig_start
                if dur.total_seconds() <= 0:
                    dur = None
            else:
                dur = None

            if default_duration_minutes is not None:
                dur = timedelta(minutes=default_duration_minutes)
            elif dur is None:
                # 명시된 end 없으면 작업 시간이 동일하다는 가정 → slot 을 duration 처럼 사용
                dur = slot

            # 시작 시간 보정: 이전 작업 시작(next_free)로부터 slot 만큼 띄우기
            if next_free is None:
                adj_start = orig_start
            else:
                adj_start = max(orig_start, next_free)

            # 다음 작업이 최소 slot 간격 확보되도록 next_free 갱신
            next_free = adj_start + slot

            adj_end = adj_start + dur

            # dict로 변환 (risk_score 포함 로직 재사용)
            d = self._task_to_dict_override(t, adj_start, adj_end)
            adjusted.append(d)

        return adjusted

    def _task_to_dict_override(self, t, start_dt, end_dt):
        """
        기존 _task_to_dict 와 동일하되 start/end 를 오버라이드하여 사용.
        """
        risk_val = None
        for name in ("risk_score", "risk", "risk_level"):
            if hasattr(t, name):
                val = getattr(t, name)
                if val is not None:
                    if isinstance(val, Decimal):
                        try:
                            val = float(val)
                        except Exception:
                            pass
                    risk_val = val
                    break

        return {
            "id": t.id,
            "status": t.status,
            "start": self._iso_local(start_dt),
            "end": self._iso_local(end_dt),
            "drain_id": t.drain_id,
            "lat": t.lat,
            "lng": t.lng,
            "risk_score": risk_val,
        }

    # ---------- GET ----------

    def get(self, request):
        date_str = request.query_params.get("date")
        include_done = str(request.query_params.get("include_done", "false")).lower() in ("1", "true", "t", "yes", "y")
        order_mode = (request.query_params.get("order") or "nearest").lower()
        by_nearest = order_mode in ("nearest", "near", "distance", "dist", "1", "true", "t", "yes", "y")

        target_date = self._parse_date(date_str)
        if target_date is None:
            return Response({"detail": "date 형식은 YYYY-MM-DD"}, status=400)

        day_start, day_end = self._bounds(target_date)
        crew = self._resolve_crew(request)

        if crew is not None:
            qs = (CleaningTask.objects
                  .filter(assigned_crew_id=crew.id,
                          scheduled_start__gte=day_start,
                          scheduled_start__lte=day_end))
            if not include_done:
                qs = qs.exclude(status="done")
            qs = self._annotate_risk(qs)

            task_list = list(qs)
            # 거리 기반 정렬 (크루에 lat/lng 필드가 있다고 가정)
            if by_nearest and getattr(crew, "lat", None) is not None and getattr(crew, "lng", None) is not None:
                task_list = self._order_by_nearest(crew.lat, crew.lng, task_list)
            else:
                task_list.sort(key=lambda t: t.scheduled_start or day_start)

            tasks = self._spread_for_display(task_list, slot_hours=2)  # ← 시작 2시간 간격 보정

            return Response({
                "date": str(target_date),
                "crew_mode": "single",
                "crew": {"id": crew.id, "name": getattr(crew, "name", str(crew.id))},
                "tasks": tasks
            }, status=200)

        # 전체 크루
        crews = list(Crew.objects.all())
        grouped = []
        for c in crews:
            qs = (CleaningTask.objects
                  .filter(assigned_crew_id=c.id,
                          scheduled_start__gte=day_start,
                          scheduled_start__lte=day_end))
            if not include_done:
                qs = qs.exclude(status="done")
            qs = self._annotate_risk(qs)

            task_list = list(qs)
            if by_nearest and getattr(c, "lat", None) is not None and getattr(c, "lng", None) is not None:
                task_list = self._order_by_nearest(c.lat, c.lng, task_list)
            else:
                task_list.sort(key=lambda t: t.scheduled_start or day_start)

            items = self._spread_for_display(task_list, slot_hours=2)  # ← 보정 적용
            grouped.append({
                "crew": {"id": c.id, "name": getattr(c, "name", str(c.id))},
                "tasks": items
            })

        return Response({
            "date": str(target_date),
            "crew_mode": "all",
            "crews_count": len(crews),
            "crews": grouped
        }, status=200)

    # ---------- POST ----------

    def post(self, request):
        """
        body: { "task_id": <int>, "when": <ISO8601>(옵션) }
        - 완료 시 SensorLogs 에 ("초음파 센서", 0.0) 1건 기록 (timestamp=현재 KST)
        """
        date_str = request.query_params.get("date")
        include_done = str(request.query_params.get("include_done", "false")).lower() in ("1", "true", "t", "yes", "y")
        order_mode = (request.query_params.get("order") or "nearest").lower()
        by_nearest = order_mode in ("nearest", "near", "distance", "dist", "1", "true", "t", "yes", "y")

        target_date = self._parse_date(date_str)
        if target_date is None:
            return Response({"detail": "date 형식은 YYYY-MM-DD"}, status=400)

        day_start, day_end = self._bounds(target_date)

        task_id = request.data.get("task_id")
        if not isinstance(task_id, int):
            return Response({"detail": "body.task_id 는 정수여야 합니다."}, status=400)

        # 완료 시각 (옵션)
        when = request.data.get("when")
        when_dt = None
        if when:
            try:
                when_dt = datetime.fromisoformat(str(when).replace("Z", "+00:00"))
                if timezone.is_naive(when_dt):
                    when_dt = timezone.make_aware(when_dt, timezone.get_current_timezone())
            except Exception:
                return Response({"detail": "when 형식은 ISO8601(YYYY-MM-DDTHH:MM[:SS][±TZ])"}, status=400)
        now = when_dt or timezone.now()

        # 날짜 범위(모든 크루)에서 task 찾기
        try:
            task = (CleaningTask.objects
                    .get(id=task_id,
                         scheduled_start__gte=day_start,
                         scheduled_start__lte=day_end))
        except CleaningTask.DoesNotExist:
            return Response({"detail": "해당 날짜 범위에서 작업을 찾을 수 없습니다."}, status=404)

        # 멱등 완료 + 센서로그 기록 (최초 완료시에만)
        if task.status != "done":
            with transaction.atomic():
                task.status = "done"
                if task.scheduled_end is None or task.scheduled_end < now:
                    task.scheduled_end = now
                task.save(update_fields=["status", "scheduled_end"])

                # ✅ 센서 로그는 '현재(Asia/Seoul) 시각'으로, 세션 타임존도 +09:00 고정
                if getattr(task, "drain_id", None) is not None:
                    kst_now = datetime.now(ZoneInfo("Asia/Seoul"))       # aware KST
                    ts_str = kst_now.strftime("%Y-%m-%d %H:%M:%S")       # 'YYYY-MM-DD HH:MM:SS'
                    with connection.cursor() as cur:
                        cur.execute("SET time_zone = '+09:00'")
                        cur.execute(
                            "INSERT INTO SensorLogs (drain_id, `timestamp`, sensor_type, value) "
                            "VALUES (%s, %s, %s, %s)",
                            [task.drain_id, ts_str, "초음파 센서", 0.0]
                        )

        # 응답 목록 구성: crew 지정 여부에 따라 단일/전체
        crew = self._resolve_crew(request)

        if crew is not None:
            qs = (CleaningTask.objects
                  .filter(assigned_crew_id=crew.id,
                          scheduled_start__gte=day_start,
                          scheduled_start__lte=day_end))
            if not include_done:
                qs = qs.exclude(status="done")
            qs = self._annotate_risk(qs)
            task_list = list(qs)

            if by_nearest and getattr(crew, "lat", None) is not None and getattr(crew, "lng", None) is not None:
                task_list = self._order_by_nearest(crew.lat, crew.lng, task_list)
            else:
                task_list.sort(key=lambda t: t.scheduled_start or day_start)

            tasks = self._spread_for_display(task_list, slot_hours=2)

            return Response({
                "date": str(target_date),
                "crew_mode": "single",
                "crew": {"id": crew.id, "name": getattr(crew, "name", str(crew.id))},
                "result": {"updated_id": task.id, "status": "done"},
                "tasks": tasks
            }, status=200)

        # 전체
        crews = list(Crew.objects.all())
        grouped = []
        for c in crews:
            qs = (CleaningTask.objects
                  .filter(assigned_crew_id=c.id,
                          scheduled_start__gte=day_start,
                          scheduled_start__lte=day_end))
            if not include_done:
                qs = qs.exclude(status="done")
            qs = self._annotate_risk(qs)

            task_list = list(qs)
            if by_nearest and getattr(c, "lat", None) is not None and getattr(c, "lng", None) is not None:
                task_list = self._order_by_nearest(c.lat, c.lng, task_list)
            else:
                task_list.sort(key=lambda t: t.scheduled_start or day_start)

            items = self._spread_for_display(task_list, slot_hours=2)

            grouped.append({
                "crew": {"id": c.id, "name": getattr(c, "name", str(c.id))},
                "tasks": items
            })

        return Response({
            "date": str(target_date),
            "crew_mode": "all",
            "crews_count": len(crews),
            "crews": grouped,
            "result": {"updated_id": task.id, "status": "done"},
        }, status=200)
