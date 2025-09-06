import math
from datetime import datetime
from django.conf import settings

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dLon/2)**2)
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def travel_minutes(a, b):
    """ a=(lat,lng), b=(lat,lng) """
    speed_kmh = getattr(settings, "SCHEDULER_AVG_SPEED_KMH", 30)
    dist_km = haversine_km(a[0], a[1], b[0], b[1])
    hours = dist_km / max(1e-6, speed_kmh)
    return int(hours * 60 + 0.5)
