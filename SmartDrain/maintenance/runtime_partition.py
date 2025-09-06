# maintenance/runtime_partition.py
from typing import List, Dict, Tuple
import math, random

def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    R = 6371.0
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(h))

def _kmeans_latlng(points: List[Tuple[float, float]], k: int, iters: int = 25, seed: int = 2025):
    if not points:
        return [], []
    random.seed(seed)
    # 초기 중심
    base = points if len(points) >= k else points * (k // max(1, len(points)) + 1)
    centroids = random.sample(base, k)[:k]
    for _ in range(iters):
        buckets = [[] for _ in range(k)]
        for idx, p in enumerate(points):
            ci = min(range(k), key=lambda j: haversine_km(p, centroids[j]))
            buckets[ci].append(idx)

        new_centroids = []
        for ci in range(k):
            if buckets[ci]:
                lat_mean = sum(points[i][0] for i in buckets[ci]) / len(buckets[ci])
                lng_mean = sum(points[i][1] for i in buckets[ci]) / len(buckets[ci])
                new_centroids.append((lat_mean, lng_mean))
            else:
                # 빈 클러스터는 기존 중심 유지
                new_centroids.append(centroids[ci])

        # 수렴 체크(아주 작은 이동이면 종료)
        moved = sum(haversine_km(centroids[i], new_centroids[i]) for i in range(k))
        centroids = new_centroids
        if moved < 1e-9:
            break

    # 최종 할당 결과
    assigns = []
    for p in points:
        ci = min(range(k), key=lambda j: haversine_km(p, centroids[j]))
        assigns.append(ci)
    return assigns, centroids

def compute_runtime_assignment(
    drains: List[Tuple[int, float, float]],  # (drain_id, lat, lng)
    crew_ids: List[int],
    seed: int
) -> Dict[int, int]:
    """
    drain_id -> crew_id 매핑을 '즉석에서' 계산 (DB에 cluster 저장 X)
    """
    mapping: Dict[int, int] = {}
    if not drains or not crew_ids:
        return mapping

    K = len(crew_ids)
    pts = [(lat, lng) for (_id, lat, lng) in drains]
    assigns, _ = _kmeans_latlng(pts, k=K, iters=25, seed=seed)

    for (drain_id, _lat, _lng), ci in zip(drains, assigns):
        mapping[drain_id] = crew_ids[ci % K]
    return mapping
