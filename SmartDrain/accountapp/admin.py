from django.contrib import admin
from .models import (
    DrainInfo, SensorLog, MotorLog,
     AlertLog, TrashStat,
    PredictionResult
)

# 전체 모델 관리자 등록 (기본 설정)
admin.site.register(DrainInfo)
admin.site.register(SensorLog)
admin.site.register(MotorLog)
admin.site.register(AlertLog)
admin.site.register(TrashStat)
admin.site.register(PredictionResult)
