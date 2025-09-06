# maintenance/runtime_assign.py
from typing import List, Tuple, Dict
from django.utils import timezone
from django.db import transaction
from datetime import datetime
from .models import CleaningTask, Crew
from .runtime_partition import compute_runtime_assignment  # 앞서 만든 유틸
from django.db.models import Q

def _tasks_in_day(target_date):
    tz = timezone.get_current_timezone()
    day_start = timezone.make_aware(datetime.combine(target_date, datetime.min.time()), tz)
    day_end   = timezone.make_aware(datetime.combine(target_date, datetime.max.time()), tz)
    return day_start, day_end

@transaction.atomic
def runtime_assign_for_date(target_date, overwrite: bool = False) -> Dict:
    """
    - 해당 날짜의 작업 중에서 배정할 대상을 모아 즉석 K-means로 crew 배정
    - overwrite=False이면 '미배정(unassigned) 작업'만 배정
    - overwrite=True이면 '그 날 모든 작업'을 재배정
    """
    day_start, day_end = _tasks_in_day(target_date)
    qs = (CleaningTask.objects
          .select_related("drain")
          .filter(predicted_due__date=target_date))  # 날짜 판단 기준은 운영 정책에 맞게 조정

    if overwrite:
        targets = list(qs)
    else:
        targets = list(qs.filter(Q(assigned_crew__isnull=True)))

    if not targets:
        return {"ok": True, "assigned": 0, "crews": 0, "detail": "no target tasks"}

    crew_ids = list(Crew.objects.values_list("id", flat=True).order_by("id"))
    if not crew_ids:
        return {"ok": False, "assigned": 0, "crews": 0, "detail": "no crews"}

    # 좌표 준비 (좌표 없는 건 스킵)
    payload: List[Tuple[int, float, float]] = []
    idx_map = {}  # task.id -> task 객체
    for t in targets:
        if t.lat is None or t.lng is None:
            continue
        payload.append((t.drain_id, float(t.lat), float(t.lng)))
        idx_map[t.drain_id] = t

    if not payload:
        return {"ok": True, "assigned": 0, "crews": len(crew_ids), "detail": "no geo tasks"}

    # 날짜 기반 seed로 재현성
    seed = int(target_date.strftime("%Y%m%d"))

    # drain_id -> crew_id
    mapping = compute_runtime_assignment(payload, crew_ids, seed=seed)

    # 매핑 반영
    assigned = 0
    for drain_id, crew_id in mapping.items():
        t = idx_map.get(drain_id)
        if not t:
            continue
        t.assigned_crew_id = crew_id
        t.save(update_fields=["assigned_crew"])
        assigned += 1

    return {"ok": True, "assigned": assigned, "crews": len(crew_ids)}
