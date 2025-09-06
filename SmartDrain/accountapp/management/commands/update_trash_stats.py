from django.core.management.base import BaseCommand
from accountapp.models import SensorLog, TrashStat, DrainInfo
from django.utils import timezone
from datetime import datetime, timedelta
from django.db.models import Avg, Max

class Command(BaseCommand):
    help = '초음파 센서 데이터를 기반으로 TrashStat 갱신'

    def add_arguments(self, parser):
        parser.add_argument(
            '--date',
            type=str,
            help='YYYY-MM-DD 형식으로 처리할 날짜를 지정 (기본값: 어제)'
        )
        parser.add_argument(
            '--drain-id',
            type=int,
            help='처리할 DrainInfo ID (필수)'
        )

    def handle(self, *args, **options):
        date_str = options.get('date')
        drain_id = options.get('drain_id')

        if not drain_id:
            self.stdout.write(self.style.ERROR("❗ 하수구 ID를 지정해주세요. 예: --drain-id=3"))
            return

        # 날짜 처리
        if date_str:
            try:
                target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                self.stdout.write(self.style.ERROR("❗ 날짜 형식이 잘못되었습니다. 예: --date=2025-08-03"))
                return
        else:
            target_date = timezone.now().date() - timedelta(days=1)

        self.stdout.write(self.style.NOTICE(f"▶️ {target_date} 날짜의 초음파 센서 데이터를 처리합니다."))

        # 날짜 범위 설정 (시간 포함)
        start_datetime = timezone.make_aware(datetime.combine(target_date, datetime.min.time()))
        end_datetime = timezone.make_aware(datetime.combine(target_date + timedelta(days=1), datetime.min.time()))

        # 센서 로그 필터링
        ultrasonic_logs = SensorLog.objects.filter(
            sensor_type='초음파',
            timestamp__gte=start_datetime,
            timestamp__lt=end_datetime
        )

        # 하수구 선택
        try:
            drain = DrainInfo.objects.get(id=drain_id)
        except DrainInfo.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"❗ ID가 {drain_id}인 하수구가 존재하지 않습니다."))
            return

        logs = ultrasonic_logs.filter(drain=drain)
        if logs.exists():
            avg_height = logs.aggregate(avg=Avg('value'))['avg']
            max_height = logs.aggregate(max=Max('value'))['max']

            stat, created = TrashStat.objects.update_or_create(
                drain=drain,
                date=target_date,
                defaults={
                    'avg_height': avg_height,
                    'max_height': max_height
                }
            )

            status = "생성됨" if created else "업데이트됨"
            self.stdout.write(self.style.SUCCESS(
                f"[{status}] {drain} - 평균: {avg_height:.2f}cm / 최대: {max_height:.2f}cm"
            ))
        else:
            self.stdout.write(self.style.WARNING(f"[데이터 없음] {drain} - 초음파 센서 로그가 없습니다."))
