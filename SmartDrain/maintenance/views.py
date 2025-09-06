#views.py
from .forecast_util import forecast_next_hours
from .serializers import CleaningTaskSerializer
from .models import CleaningTask, Crew
from .services import generate_tasks_for_all_drains
from .scheduler import greedy_schedule_for_date
from rest_framework.views import APIView
from accountapp.models import DrainInfo  # ★ 교체
from .runtime_assign import runtime_assign_for_date  # ← 추가 import
from rest_framework.views import APIView
from rest_framework.response import Response
from datetime import datetime
from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from datetime import datetime
from django.utils import timezone
from django.db import transaction
   # ← generate 함수가 있는 모듈명에 맞춰 수정

# views.py (혹은 해당 View가 있는 파일)

from django.db import transaction
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.utils import timezone
from datetime import datetime

#from maintenance.scheduler import greedy_schedule_for_date
#from maintenance.models import CleaningTask, DrainInfo  # 경로는 프로젝트 구조에 맞게 import
# from .predictor import generate_tasks_for_all_drains  # 이미 쓰던 함수 import

from datetime import datetime, time, timedelta
from django.db import transaction
from django.db.models import DateTimeField, DateField
from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

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

        # ===== 1) 예측 → 임계 도달 계산 → CleaningTask 생성 =====
        # 생성 전/후 ID 차집합으로 '이번 요청에서 새로 만든 작업'만 정확히 집어냄
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

        # ===== 1.5) 스케줄 중복 방지 (같은 날짜·같은 하수구 중복 배정 금지) =====
        # 새로 만든 task들의 drain 매핑
        task_rows = CleaningTask.objects.filter(id__in=created_ids).values("id", "drain_id")
        task_to_drain = {r["id"]: r["drain_id"] for r in task_rows}
        drains_in_created = set(task_to_drain.values())

        # CleaningTask 내부에서 "당일에 이미 스케줄된" 동일 하수구가 있는지 확인
        date_filter = _build_ct_schedule_date_filter(target_date)

        existing_qs = CleaningTask.objects.filter(drain_id__in=drains_in_created)
        if date_filter:
            existing_qs = existing_qs.filter(**date_filter)

        # (필요 시 상태 제외: 예, 완료/취소 제외)
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
                # 방금 생성된 것까지 혹시 잡히지 않도록 제외
                del_qs = del_qs.exclude(id__in=created_ids)
                del_qs.delete()
            filtered_ids = list(created_ids)

        body["schedule_guard"] = {
            "overwrite": overwrite,
            "already_scheduled_drains": list(already_scheduled_drains),
            "skipped_task_ids": skipped_task_ids,
            "scheduled_task_candidates": filtered_ids,
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
                task_ids=filtered_ids,          # ★ 핵심: 중복 제거된 목록만 스케줄링
                assign_all=False,
                allow_overtime=False,
                force_assign_remaining=True,   # 생성분을 근무시간/윈도우로 못 넣으면 2차에서라도 넣고 싶으면 True
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

        # ===== 3) (옵션) 예측 응답 포함하려면 여기에서 include_forecast/horizon_hours 활용 =====

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


from datetime import datetime
from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
#from maintenance.models import CleaningTask


# views.py
from datetime import datetime
from django.utils import timezone
from django.db import transaction
from rest_framework.views import APIView
from rest_framework.response import Response

# CleaningTask, Crew 는 이 뷰가 속한 앱에 있다고 가정 (필요시 경로 조정)
from .models import Crew, CleaningTask

# SensorLog 는 "다른 앱"에 있음 (예: smartdrain 앱) → 실제 경로에 맞게 수정하세요.
from accountapp.models import SensorLog


from datetime import datetime
from django.utils import timezone
from django.db import transaction
from rest_framework.views import APIView
from rest_framework.response import Response

# 프로젝트 경로에 맞게 조정
from .models import Crew, CleaningTask
from accountapp.models import SensorLog   # SensorLog가 있는 앱 경로

from datetime import datetime
from django.utils import timezone
from django.db import transaction, connection   # ★ connection 추가
from rest_framework.views import APIView
from rest_framework.response import Response

from .models import Crew, CleaningTask
# SensorLog ORM은 사용하지 않으므로 import 불필요 (원하면 남겨도 무방)

from datetime import datetime
from zoneinfo import ZoneInfo

from django.utils import timezone
from django.db import transaction, connection
from rest_framework.views import APIView
from rest_framework.response import Response

# 프로젝트 경로에 맞게 조정
from .models import Crew, CleaningTask


from datetime import datetime, date as date_cls, time, timedelta  # ⬅️ date_cls 추가 임포트

from datetime import datetime, date as date_cls, time, timedelta
from zoneinfo import ZoneInfo
from decimal import Decimal

from django.apps import apps
from django.conf import settings
from django.db import transaction, connection
from django.db.models import OuterRef, Subquery, DateTimeField, DateField
from django.utils import timezone

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

# 실제 경로에 맞게 조정하세요.
# 예: from core.models import CleaningTask, Crew
from .models import CleaningTask, Crew


from datetime import datetime, date as date_cls, time, timedelta
from zoneinfo import ZoneInfo
from decimal import Decimal

from django.apps import apps
from django.conf import settings
from django.db import transaction, connection
from django.db.models import OuterRef, Subquery, DateTimeField, DateField
from django.utils import timezone

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

# 실제 경로에 맞게 조정하세요.
# 예: from core.models import CleaningTask, Crew
from .models import CleaningTask, Crew


class CrewTasksView(APIView):
    """
    GET  /api/crews/tasks?date=YYYY-MM-DD&include_done=false[&crew=팀A|all][&crew_id=1]
         - crew/crew_id 생략 또는 crew=all 이면 모든 크루를 그룹핑해서 반환

    POST /api/crews/tasks?date=YYYY-MM-DD
         body: { "task_id": 222, "when": "2025-09-02T10:20:00+09:00"(옵션) }
         - crew 미지정이어도 해당 날짜 범위에서 task_id 를 찾아 done 처리
         - ✅ done 처리 시, 해당 하수구 센서 로그(SensorLogs)에 ("초음파 센서", 0.0) 1건 추가
           (센서 로그의 timestamp는 항상 '현재 KST'로 저장, 세션 타임존 +09:00 고정)
         - 응답 task 항목에만 risk_score 포함 (요약 X)
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

    # ---------- GET ----------
    def get(self, request):
        date_str = request.query_params.get("date")
        include_done = str(request.query_params.get("include_done", "false")).lower() in ("1", "true", "t", "yes", "y")

        target_date = self._parse_date(date_str)
        if target_date is None:
            return Response({"detail": "date 형식은 YYYY-MM-DD"}, status=400)

        day_start, day_end = self._bounds(target_date)
        crew = self._resolve_crew(request)

        if crew is not None:
            qs = (CleaningTask.objects
                  .filter(assigned_crew_id=crew.id,
                          scheduled_start__gte=day_start,
                          scheduled_start__lte=day_end)
                  .order_by("scheduled_start"))
            if not include_done:
                qs = qs.exclude(status="done")
            qs = self._annotate_risk(qs)  # 필요 시 risk_score 주입

            data = [self._task_to_dict(t) for t in qs]  # 각 task dict 안에만 risk_score 존재

            return Response({
                "date": str(target_date),
                "crew_mode": "single",
                "crew": {"id": crew.id, "name": getattr(crew, "name", str(crew.id))},
                "tasks": data
            }, status=200)

        # 전체 크루
        crews = list(Crew.objects.all())
        grouped = []
        for c in crews:
            qs = (CleaningTask.objects
                  .filter(assigned_crew_id=c.id,
                          scheduled_start__gte=day_start,
                          scheduled_start__lte=day_end)
                  .order_by("scheduled_start"))
            if not include_done:
                qs = qs.exclude(status="done")
            qs = self._annotate_risk(qs)

            items = [self._task_to_dict(t) for t in qs]  # 각 task dict 안에만 risk_score 존재
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
                          scheduled_start__lte=day_end)
                  .order_by("scheduled_start"))
            if not include_done:
                qs = qs.exclude(status="done")
            qs = self._annotate_risk(qs)  # 필요 시 risk_score 주입
            tasks = [self._task_to_dict(t) for t in qs]
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
                          scheduled_start__lte=day_end)
                  .order_by("scheduled_start"))
            if not include_done:
                qs = qs.exclude(status="done")
            qs = self._annotate_risk(qs)  # 필요 시 risk_score 주입
            items = [self._task_to_dict(t) for t in qs]
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
