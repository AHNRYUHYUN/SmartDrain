from celery import shared_task
from .management.commands.update_trash_stats import Command

@shared_task
def update_trash_stats_task():
    Command().handle()