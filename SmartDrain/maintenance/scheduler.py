# maintenance/scheduler.py
from datetime import datetime, timedelta, time
from django.utils import timezone
from django.db import models
from .models import CleaningTask, Crew
from .utils import travel_minutes


def _parse_now_override(now_override, tz):
    """ 문자열/naive/aware datetime 모두 안전 처리해서 aware로 반환 """
    if now_override is None:
        return timezone.now()
    if isinstance(now_override, str):
        # "YYYY-MM-DDTHH:MM" 또는 "YYYY-MM-DD HH:MM:SS" 등
        dt = datetime.fromisoformat(now_override.strip())
    elif isinstance(now_override, datetime):
        dt = now_override
    else:
        return timezone.now()
    if timezone.is_naive(dt):
        return timezone.make_aware(dt, tz)
    return dt.astimezone(tz)


def greedy_schedule_for_date(
    target_date=None,
    overwrite=False,              # 오늘 기존 배정 초기화
    assign_all=False,             # (1차) window 제약 무시
    allow_overtime=False,         # (1차) 근무시간 초과 허용
    balance_by_minutes=True,      # 균형 기준: 누적 분(True) / 건수(False)
    min_gap_min=120,                # 작업 간 최소 간격(분)
    lunch_break=None,             # ("12:00","13:00") or None
    force_assign_remaining=True,  # (2차) 남은 작업 강제 배정(야근 허용)
    dry_run=False,                # DB 미변경 미리보기
    task_ids=None,                # ← 이 ID들만 대상으로(선택)
    created_only_today=False,     # ← CleaningTask에 created_at 있으면 오늘 생성분만(선택)
    use_current_time=False,       # ★ 오늘이면 '지금 이후'만 배정
    now_override=None,            # ★ 문자열/datetime로 '지금' 고정(데모/리플레이)
):
    """
    오늘 '대상 집합'만 스케줄링:
      - 1차: 대상 집합 중 window/근무/이동/간격/점심을 지키며 최대 배정
      - 2차: 1차에서 못 넣은 '대상 집합'의 남은 것만 강제 배정(야근 허용)
    옵션:
      - use_current_time=True → target_date가 오늘이면 현재시각 이후만 배정
      - now_override="2025-09-17T13:00" → 데모용으로 '지금'을 고정
    """
    tz = timezone.get_current_timezone()
    if target_date is None:
        target_date = timezone.localdate()

    # '지금' 앵커 계산 (override 지원)
    now_anchor = _parse_now_override(now_override, tz)

    day_start = timezone.make_aware(datetime.combine(target_date, datetime.min.time()), tz)
    day_end   = timezone.make_aware(datetime.combine(target_date, datetime.max.time()), tz)

    # 점심시간 블록(옵션)
    lb_start = lb_end = None
    if lunch_break and isinstance(lunch_break, (tuple, list)) and len(lunch_break) == 2:
        h1, m1 = map(int, lunch_break[0].split(":"))
        h2, m2 = map(int, lunch_break[1].split(":"))
        lb_start = timezone.make_aware(datetime.combine(target_date, time(h1, m1)), tz)
        lb_end   = timezone.make_aware(datetime.combine(target_date, time(h2, m2)), tz)

    crews = list(Crew.objects.all())
    crew_by_id = {c.id: c for c in crews}

    # (옵션) 기존 배정 초기화
    if overwrite and not dry_run:
        (CleaningTask.objects
         .filter(status="scheduled", scheduled_start__gte=day_start, scheduled_start__lte=day_end)
         .update(assigned_crew=None, scheduled_start=None, scheduled_end=None, status="pending"))

    # ===== 후보 집합 구성 =====
    status_q = (models.Q(status="pending") |
                models.Q(status="scheduled", scheduled_start__isnull=True))

    if task_ids:
        # 명시된 ID만(윈도우 필터 생략)
        base_qs = CleaningTask.objects.filter(status_q, id__in=list(task_ids))
    else:
        base_qs = CleaningTask.objects.filter(status_q)

        # created_only_today 우선
        if created_only_today and hasattr(CleaningTask, "created_at"):
            base_qs = base_qs.filter(created_at__gte=day_start, created_at__lte=day_end)
        else:
            # 기본: 오늘 윈도우와 겹치는 작업
            base_qs = (base_qs
                .filter(models.Q(window_end__isnull=True) | models.Q(window_end__gte=day_start))
                .filter(models.Q(window_start__isnull=True) | models.Q(window_start__lte=day_end))
            )

        # ★ 오늘 + use_current_time이면, '지금 이후'만 남기기 (window_end < now는 제외)
        if use_current_time and (now_anchor.date() == target_date):
            base_qs = base_qs.filter(models.Q(window_end__isnull=True) | models.Q(window_end__gte=now_anchor))

    # 이미 오늘 배정된 것은 후보에서 제외(중복 가드)
    scheduled_today_ids = set(
        CleaningTask.objects
        .filter(status="scheduled", scheduled_start__gte=day_start, scheduled_start__lte=day_end)
        .values_list("id", flat=True)
    )
    all_tasks = {t.id: t for t in base_qs if t.id not in scheduled_today_ids}

    unassigned = dict(all_tasks)
    schedules = {c.id: [] for c in crews}
    scheduled_ids = set(scheduled_today_ids)

    # 크루 상태: 오늘 + use_current_time이면 시작시각을 max(shift_start, now_anchor)로
    crew_state = {}
    for c in crews:
        sdt = timezone.make_aware(datetime.combine(target_date, c.shift_start), tz)
        if use_current_time and (now_anchor.date() == target_date):
            sdt = max(sdt, now_anchor)
        edt = timezone.make_aware(datetime.combine(target_date, c.shift_end), tz)
        crew_state[c.id] = {
            "cur_pos": (c.home_lat, c.home_lng),
            "cur_time": sdt,
            "end_time": edt,
            "assigned_minutes": 0,
        }

    def _apply_lunch_block(start, dur_td):
        if lb_start is None or lb_end is None:
            return start
        finish = start + dur_td
        if (start < lb_end) and (finish > lb_start):
            return max(start, lb_end)
        return start

    # ===== 1) 규칙 준수 글로벌 그리디 =====
    while unassigned:
        best = None  # (priority, crew_id, task, start, finish)

        for c in crews:
            st = crew_state[c.id]["cur_time"]
            et = crew_state[c.id]["end_time"]
            if (st >= et) and (not allow_overtime):
                continue
            cur_pos = crew_state[c.id]["cur_pos"]
            bal_key = crew_state[c.id]["assigned_minutes"] if balance_by_minutes else len(schedules[c.id])
            best_for_crew = None

            for tid, t in list(unassigned.items()):
                if t.id in scheduled_ids:
                    unassigned.pop(tid, None)
                    continue

                move_min = int(travel_minutes(cur_pos, (t.lat, t.lng)))
                arrive   = st + timedelta(minutes=move_min)
                dur_min  = int(getattr(t, "estimated_duration_min", 0) or 30)
                dur_td   = timedelta(minutes=dur_min)
                wstart, wend = t.window_start, t.window_end

                # 시작 전 최소 간격 + (오늘이면) 지금 이후 보장
                start = max(arrive, st + timedelta(minutes=min_gap_min))
                if use_current_time and (now_anchor.date() == target_date):
                    start = max(start, now_anchor)

                if not assign_all and wstart:
                    # 윈도우 시작도 존중
                    start = max(start, wstart)

                start = _apply_lunch_block(start, dur_td)
                finish = start + dur_td

                # 근무시간/윈도우/오버타임 검증
                if (not allow_overtime) and (finish > et):
                    continue
                if (not assign_all) and (wend is not None) and (finish > wend):
                    continue

                risk = float(getattr(t, "risk_score", 0.0) or 0.0)
                wnd_end_for_sort = wend or timezone.make_aware(datetime.max.replace(year=9999), tz)
                priority = (bal_key, finish, move_min, wnd_end_for_sort, -risk)
                cand = (priority, c.id, t, start, finish)
                if (best_for_crew is None) or (cand[0] < best_for_crew[0]):
                    best_for_crew = cand

            if best_for_crew is not None:
                if (best is None) or (best_for_crew[0] < best[0]):
                    best = best_for_crew

        if best is None:
            break

        _, crew_id, task, st_dt, ft_dt = best
        if not dry_run:
            task.assigned_crew   = crew_by_id[crew_id]
            task.scheduled_start = st_dt
            task.scheduled_end   = ft_dt
            task.status          = "scheduled"
            task.save(update_fields=["assigned_crew", "scheduled_start", "scheduled_end", "status"])

        scheduled_ids.add(task.id)
        unassigned.pop(task.id, None)
        schedules[crew_id].append((task.id, st_dt, ft_dt))
        crew_state[crew_id]["cur_pos"]  = (task.lat, task.lng)
        crew_state[crew_id]["cur_time"] = ft_dt
        crew_state[crew_id]["assigned_minutes"] += int((ft_dt - st_dt).total_seconds() // 60)

    # ===== 2) 남은 작업 강제 배정(옵션) =====
    if force_assign_remaining and unassigned:
        far_future = timezone.make_aware(datetime.max.replace(year=9999), tz)
        remaining = [t for t in unassigned.values() if t.id not in scheduled_ids]
        remaining.sort(key=lambda t: (
            t.window_end or far_future,
            -float(getattr(t, "risk_score", 0.0) or 0.0),
            t.id,
        ))
        for t in remaining:
            best_force = None  # (key, crew_id, start, finish)
            for c in crews:
                st = crew_state[c.id]["cur_time"]
                cur_pos = crew_state[c.id]["cur_pos"]
                dur_min = int(getattr(t, "estimated_duration_min", 0) or 30)
                move_min = int(travel_minutes(cur_pos, (t.lat, t.lng)))

                start = st + timedelta(minutes=max(move_min, min_gap_min))
                if use_current_time and (now_anchor.date() == target_date):
                    start = max(start, now_anchor)

                start = _apply_lunch_block(start, timedelta(minutes=dur_min))
                finish = start + timedelta(minutes=dur_min)

                key = (finish, crew_state[c.id]["assigned_minutes"], move_min, len(schedules[c.id]))
                cand = (key, c.id, start, finish)
                if (best_force is None) or (cand[0] < best_force[0]):
                    best_force = cand

            _, crew_id, st_dt, ft_dt = best_force
            if not dry_run:
                t.assigned_crew   = crew_by_id[crew_id]
                t.scheduled_start = st_dt
                t.scheduled_end   = ft_dt
                t.status          = "scheduled"
                t.save(update_fields=["assigned_crew", "scheduled_start", "scheduled_end", "status"])
            scheduled_ids.add(t.id)
            schedules[crew_id].append((t.id, st_dt, ft_dt))
            crew_state[crew_id]["cur_pos"]  = (t.lat, t.lng)
            crew_state[crew_id]["cur_time"] = ft_dt
            crew_state[crew_id]["assigned_minutes"] += int((ft_dt - st_dt).total_seconds() // 60)

    # 반환 전, 혹시 모를 중복 clean
    for cid, evs in list(schedules.items()):
        seen, clean = set(), []
        for tid, st, ft in evs:
            if tid in seen:
                continue
            seen.add(tid)
            clean.append((tid, st, ft))
        schedules[cid] = clean

    return schedules
