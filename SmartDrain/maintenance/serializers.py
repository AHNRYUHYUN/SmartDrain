from rest_framework import serializers
from .models import Crew, CleaningTask

class CrewSerializer(serializers.ModelSerializer):
    class Meta:
        model = Crew
        fields = \
            "__all__"

class CleaningTaskSerializer(serializers.ModelSerializer):
    class Meta:
        model = CleaningTask
        fields = \
            "__all__"
