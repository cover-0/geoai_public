"""
문제 3 (4장). 새 지역에서도 이 점수가 나오는가
===============================================
500개 관측점(spatial_points.geojson) 가운데 학번으로 400개를 뽑아,
접근성 지표(target)를 예측하는 랜덤 포레스트의 R2를 보고한다.
보고서의 문장: "이 모델은 아직 조사하지 않은 새 지역에서도 이 점수를 낸다."

피처 사전
  x_coord, y_coord     관측점 좌표(m)
  dist_center_0..2     관심 지점 3곳까지 거리(m)
  survey_score         사후 만족도 조사 점수 -- 지표가 확정된 뒤 현장 조사로 얻음

실행: python exam/q3.py --id 학번
"""

import geopandas as gpd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold, cross_val_score

from common import PRACTICE, key, parse_id, rng_of

sid = parse_id()
rng = rng_of(sid, 3)
gdf = gpd.read_file(PRACTICE / "chapter4" / "data" / "spatial_points.geojson")
sub = gdf.iloc[rng.choice(len(gdf), 400, replace=False)].reset_index(drop=True)
sub["survey_score"] = sub["target"].values + rng.normal(0, 3, len(sub))

centers = np.array([[320000, 4165000], [315000, 4155000], [330000, 4170000]])
x, y = sub["x_coord"].values, sub["y_coord"].values
for i, (cx, cy) in enumerate(centers):
    sub[f"dist_center_{i}"] = np.hypot(x - cx, y - cy)

# [수정 1] 목표값 누수: survey_score는 지표(target)가 확정된 뒤 현장 조사로 얻는 값이라
# 새 지역을 예측하는 시점에는 알 수 없다(27행에서 target + 잡음으로 만들어짐). 피처에서 뺀다.
features = ["x_coord", "y_coord", "dist_center_0", "dist_center_1", "dist_center_2"]
X, t = sub[features].values, sub["target"].values
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, t, cv=cv, scoring="r2")

# [수정 2] 공간 분할 누수: 무작위 분할은 가까운 관측점을 학습과 검증에 나눠 넣어
# '새 지역'이 아니라 '본 지역 사이 빈 곳 메우기' 성능을 잰다. 4장 7.2절대로
# 좌표를 5km 격자 블록으로 잘라 같은 블록은 같은 폴드에 넣는다(GroupKFold 5).
block_size = 5000
block_id = ((sub.geometry.x // block_size).astype(int) * 10000
            + (sub.geometry.y // block_size).astype(int)).values
block_scores = cross_val_score(model, X, t, cv=GroupKFold(n_splits=5), groups=block_id, scoring="r2")

r2_random = float(scores.mean())
r2_block = float(block_scores.mean())
diff = r2_random - r2_block
block_sd = float(block_scores.std())
# 4장 7절: 재려는 차이가 폴드 간 표준편차보다 작으면 그 차이를 결과로 인용하지 않는다
cite_diff = 1 if diff > block_sd else 0

print(f"관측점 {len(sub)}개, 피처 {features}")
print(f"[무작위 5겹] R2: {np.round(scores, 3).tolist()}  평균 {r2_random:.3f}")
print(f"[5km 블록 {len(np.unique(block_id))}개, GroupKFold 5] R2: {np.round(block_scores, 3).tolist()}  "
      f"평균 {r2_block:.3f}, 폴드 표준편차 {block_sd:.3f}")
print(f"무작위 - 블록 차이 = {diff:.3f} ({'표준편차보다 큼: 인용 가능' if cite_diff else '표준편차 이하: 인용하지 않음'})")
key("r2_random", r2_random)
key("r2_block", r2_block)
key("diff", diff)
key("block_sd", block_sd)
key("cite_diff", cite_diff)
