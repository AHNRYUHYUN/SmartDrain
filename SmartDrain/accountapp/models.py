from django.db import models

# 하수구 정보 모델
class DrainInfo(models.Model):
    region = models.CharField(max_length=100, verbose_name="시/도")  # 서울특별시
    sub_region = models.CharField(max_length=100, verbose_name="구")  # 중구
    name = models.CharField(max_length=100, verbose_name="하수구 이름 또는 번호")  # A10번 하수구
    latitude = models.FloatField(verbose_name="위도")
    longitude = models.FloatField(verbose_name="경도")

    class Meta:
        db_table = 'DrainInfos'
        verbose_name = '하수구 정보'
        verbose_name_plural = '하수구 정보 목록'
        unique_together = ('region', 'sub_region','name')

    def __str__(self):
        return f"{self.region} {self.sub_region}  {self.name}"




# 센서 로그
class SensorLog(models.Model):
    drain = models.ForeignKey(DrainInfo, on_delete=models.CASCADE, verbose_name="하수구 정보")
    timestamp = models.DateTimeField( null=True,verbose_name="측정 시간")
    sensor_type = models.CharField(max_length=50, verbose_name="센서 유형")
    value = models.FloatField(verbose_name="측정 값 (cm, 유무 등)")

    class Meta:
        db_table = 'SensorLogs'
        verbose_name = '센서 로그'
        verbose_name_plural = '센서 로그 목록'
        ordering = ['-timestamp']

    def __str__(self):
        return f"[{self.timestamp}] {self.sensor_type} @ {self.drain} = {self.value}"

from django.db import models
from django.utils import timezone

# DrainInfo 는 이미 있으시다고 하셔서 참조만 합니다.
# 같은 app 안에 있다면 문자열로 참조해 순환 import를 피합니다.

from django.db import models
from django.utils import timezone

class RainLog(models.Model):
    drain = models.ForeignKey(
        'DrainInfo',  # 문자열로 참조 가능, app_label.DrainInfo 형태도 가능
        on_delete=models.CASCADE,
        verbose_name="하수구 정보"
    )
    # 요청 시각을 기본값으로 현재 시간으로 자동 저장
    timestamp = models.DateTimeField(
        verbose_name="측정 시간",
        default=timezone.now,
        null=True,
        db_index=True
    )
    value = models.FloatField(
        verbose_name="측정 값 (cm, 유무 등)"
    )

    class Meta:
        db_table = 'rain'
        verbose_name = '센서 로그'
        verbose_name_plural = '센서 로그 목록'
        ordering = ['-timestamp']

    def __str__(self):
        return f"[{self.timestamp:%Y-%m-%d %H:%M:%S}] drain={self.drain_id} value={self.value}"


# 경고 로그
class AlertLog(models.Model):
    drain = models.ForeignKey(DrainInfo, on_delete=models.CASCADE, verbose_name="하수구 정보")
    timestamp = models.DateTimeField(verbose_name="발송 시각")
    alert_type = models.CharField(max_length=50, verbose_name="경고 유형")
    recipient = models.CharField(max_length=100, verbose_name="수신자(관리자) 정보")

    class Meta:
        db_table = 'AlertLogs'
        verbose_name = '경고 로그'
        verbose_name_plural = '경고 로그 목록'
        ordering = ['-timestamp']

    def __str__(self):
        return f"[{self.timestamp}] {self.alert_type} alert sent to {self.recipient} at {self.drain}"


# 쓰레기 통계
class TrashStat(models.Model):
    drain = models.ForeignKey(DrainInfo, on_delete=models.CASCADE, verbose_name="하수구 정보")
    date = models.DateField(verbose_name="날짜")
    avg_height = models.FloatField(verbose_name="평균 쓰레기 높이 (cm)")
    max_height = models.FloatField(verbose_name="최고 쓰레기 적재량 (cm)")

    class Meta:
        db_table = 'TrashStats'
        verbose_name = '쓰레기 통계'
        verbose_name_plural = '쓰레기 통계 목록'
        ordering = ['-date']

    def __str__(self):
        return f"[{self.date}] {self.drain}: Avg {self.avg_height}cm / Max {self.max_height}cm"


# 예측 결과
class PredictionResult(models.Model):
    drain = models.ForeignKey(DrainInfo, on_delete=models.CASCADE, verbose_name="하수구 정보")
    date = models.DateTimeField(verbose_name="예측 시각")
    predicted_risk_level = models.CharField(max_length=20, verbose_name="예측 위험도 (low, moderate, high)")
    predicted_value = models.FloatField(verbose_name="예측 쓰레기 높이 (cm)")

    class Meta:
        db_table = 'PredictionResults'
        verbose_name = '예측 결과'
        verbose_name_plural = '예측 결과 목록'
        ordering = ['-date']

    def __str__(self):
        return f"[{self.date}] {self.drain}: Risk={self.predicted_risk_level}, Trash={self.predicted_value}cm"
