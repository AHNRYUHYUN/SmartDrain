# maintenance/models.py

from django.db import models

class Crew(models.Model):
    name = models.CharField(max_length=50)
    home_lat = models.FloatField()
    home_lng = models.FloatField()
    shift_start = models.TimeField(default="09:00")
    shift_end = models.TimeField(default="18:00")
    capacity_per_day = models.IntegerField(default=20)


    def __str__(self):
        return self.name

class CleaningTask(models.Model):
    STATUS_CHOICES = (
        ("pending", "Pending"),
        ("scheduled", "Scheduled"),
        ("done", "Done"),
    )
    # ❗여기를 'accountapp.DrainInfo'로 교체
    drain = models.ForeignKey(
        'accountapp.DrainInfo',  # ← 변경!
        on_delete=models.CASCADE,
        related_name='tasks'
    )
    lat = models.FloatField()
    lng = models.FloatField()
    predicted_due = models.DateTimeField()
    estimated_duration_min = models.IntegerField(default=20)
    risk_score = models.FloatField(default=0.0)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    window_start = models.DateTimeField(null=True, blank=True)
    window_end   = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    assigned_crew = models.ForeignKey('maintenance.Crew', null=True, blank=True, on_delete=models.SET_NULL,
                                      related_name='assigned_tasks')
    scheduled_start = models.DateTimeField(null=True, blank=True)
    scheduled_end = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Task(drain={self.drain_id}, due={self.predicted_due}, status={self.status})"
