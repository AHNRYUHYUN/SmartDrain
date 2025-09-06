from django.urls import path, include
from rest_framework.routers import DefaultRouter
from django.urls import path
from .views import update_trash_stats_form, SensorLogsByDrainInfoAPIView, TrainImprovedDrainModelsView,PredictDrainStatusView
from .views import (
    SensorLogViewSet, MotorLogViewSet,
    AlertLogViewSet, TrashStatViewSet, PredictionResultViewSet,  DrainInfoViewSet,
    update_trash_stats_view
)

router = DefaultRouter()
router.register(r'sensorlogs', SensorLogViewSet)
router.register(r'motorlogs', MotorLogViewSet)
router.register(r'alertlogs', AlertLogViewSet)
router.register(r'trashstats', TrashStatViewSet)
router.register(r'predictionresults', PredictionResultViewSet)
router.register(r'drains', DrainInfoViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('update-trash-stats-form/', update_trash_stats_form, name='update_trash_stats_form'),
    path('sensorvalue/', SensorLogsByDrainInfoAPIView.as_view(), name='sensor-logs-by-draininfo'),
    path('train-model/', TrainImprovedDrainModelsView.as_view(), name='train_lstm_model'),
    #모델 만드는 코드
    path("drainpredict/", PredictDrainStatusView.as_view(), name="drainpredict"),
]
