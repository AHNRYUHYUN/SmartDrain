from django.urls import path
from django.urls import path, include
from .views import (
    PredictAndGenerateTasksView,
    TaskListView,
    DaySchedulePreviewView, PredictDebugView, DayScheduleDetailedView, DayRouteGeoJSONView,
    CrewTasksView,
)



urlpatterns = [
    # 모든 배수구에 대해 예측을 수행하고, 청소 필요 시점을 기준으로 CleaningTask 생성
    path('predict-and-generate-tasks/', PredictAndGenerateTasksView.as_view()),


    # CleaningTask 목록 조회
    # ?status=pending|scheduled|done 쿼리로 상태별 필터링 가능
    path('tasks/', TaskListView.as_view()),

    # 지정한 날짜(기본: 오늘)에 대해 스케줄을 미리 계산해 보는 기능 (상태 변경 없음)
    path('schedules/day/', DaySchedulePreviewView.as_view()),

    path('debug/predict/', PredictDebugView.as_view()),  # 디버그

    path('schedules/day/detailed/', DayScheduleDetailedView.as_view()),

    # ★ crew별 경로(라인)와 정류장 정보(스탑)를 GeoJSON으로 반환 (프론트에서 지도에 그림)
    path('schedules/day/route/', DayRouteGeoJSONView.as_view()),

    path("crews/tasks", CrewTasksView.as_view(), name="crew-tasks"),  # GET(전체/단일) + POST(완료)

    path('api/accountapp/', include('accountapp.urls')),
]




