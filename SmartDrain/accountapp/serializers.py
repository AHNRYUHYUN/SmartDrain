from rest_framework import serializers
from .models import SensorLog, MotorLog, AlertLog, TrashStat, PredictionResult, DrainInfo


class DrainInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = DrainInfo
        fields =\
            '__all__'
class SensorLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = SensorLog
        fields = \
            '__all__'


class MotorLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = MotorLog
        fields = \
            '__all__'




class AlertLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = AlertLog
        fields = \
            '__all__'


class TrashStatSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrashStat
        fields = \
            '__all__'


class PredictionResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictionResult
        fields = \
            '__all__'
