from rest_framework import serializers
from .models import SensorLog,  AlertLog, TrashStat, PredictionResult, DrainInfo


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



from rest_framework import serializers
from django.utils import timezone
from .models import RainLog

class RainLogSerializer(serializers.ModelSerializer):
    # timestamp를 읽기 전용으로 해서 클라이언트가 값 못 넣게 하고,
    # 생성 시에는 서버 시간(now)으로 저장되도록 합니다.
    timestamp = serializers.DateTimeField(read_only=True)

    class Meta:
        model = RainLog
        fields = ['id', 'drain', 'timestamp', 'value']

    def create(self, validated_data):
        validated_data['timestamp'] = timezone.now()
        return super().create(validated_data)




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
