from django.contrib import admin
from .models import Crew, CleaningTask

@admin.register(Crew)
class CrewAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "home_lat", "home_lng", "shift_start", "shift_end", "capacity_per_day")

@admin.register(CleaningTask)
class CleaningTaskAdmin(admin.ModelAdmin):
    list_display = ("id", "drain", "predicted_due", "risk_score", "status", "window_start", "window_end")
    list_filter = ("status",)
    search_fields = ("drain__id",)
