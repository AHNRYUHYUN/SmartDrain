from django.urls import path, include
from rest_framework.routers import DefaultRouter
from django.urls import path
from .views import update_trash_stats_form, SensorLogsByDrainInfoAPIView, TrainImprovedDrainModelsView, \
    PredictDrainStatusView, RainLogViewSet
from .views import (
    SensorLogViewSet,

    AlertLogViewSet, TrashStatViewSet, PredictionResultViewSet,  DrainInfoViewSet,
)

router = DefaultRouter()
router.register(r'sensorlogs', SensorLogViewSet)
router.register(r'alertlogs', AlertLogViewSet)
router.register(r'trashstats', TrashStatViewSet)
router.register(r'predictionresults', PredictionResultViewSet)
router.register(r'drains', DrainInfoViewSet)
router.register(r'rain-logs', RainLogViewSet)


urlpatterns = [
    path('', include(router.urls)),
    path('update-trash-stats-form/', update_trash_stats_form, name='update_trash_stats_form'),
    path('sensorvalue/', SensorLogsByDrainInfoAPIView.as_view(), name='sensor-logs-by-draininfo'),
    path('train-model/', TrainImprovedDrainModelsView.as_view(), name='train_lstm_model'),
    #모델 만드는 코드
    path("drainpredict/", PredictDrainStatusView.as_view(), name="drainpredict"),
]
