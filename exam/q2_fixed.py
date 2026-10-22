"""
문제 2 (3장). 지도에 0개로 나온 업종을 비었다고 판정해도 되는가
=================================================================
서울 412개 동에서, 상가정보(등록부)에 있는 점포 중 지도(OSM)에 나타난 비율이 포착률이다.
다섯 업종의 포착률과 동별 점포 수는 학번으로 정해지는 모의 자료다.

실행: python exam/q2.py --id 학번
"""

import numpy as np

from common import key, parse_id, rng_of

KINDS = ["서점", "안경", "화장품", "약국", "카페"]
N_DONG = 412

sid = parse_id()
rng = rng_of(sid, 2)
lam = rng.uniform(0.3, 3.5, len(KINDS))
registry = [rng.negative_binomial(1.5, 1.5 / (1.5 + l), N_DONG) for l in lam]  # 동별 등록 점포 수
p_true = rng.uniform(0.04, 0.6, len(KINDS))
over = int(rng.integers(0, len(KINDS)))          # 대응표 문제가 있는 업종 하나
osm_total = []
for k, reg in enumerate(registry):
    r = int(reg.sum())
    osm_total.append(int(r * rng.uniform(1.03, 1.15)) if k == over else int(rng.binomial(r, p_true[k])))

CUT_BIZ, CUT_PUB = 0.80, 0.20

print(f"{'업종':<6}{'등록':>7}{'지도':>7}{'포착률':>8}{'0개 동 비율':>12}{'사후확률':>10}  판정")
for k, name in enumerate(KINDS):
    reg = registry[k]
    p = osm_total[k] / int(reg.sum())
    prior0 = float((reg == 0).mean())
    # [수정 2] 포착률은 확률이라 1을 넘을 수 없다. 1을 넘으면 등록부-지도 대응표가 어긋난 것이므로
    # (1-p)가 음수가 되는 계산을 하지 않고 사후확률은 NA, 판정은 '대응표 점검'으로 뺀다.
    if p > 1:
        posterior = None
        verdict = "대응표 점검"
    else:
        # [수정 1] 0개 동 비율은 사전확률 P(N=0)일 뿐, 지도에 0개로 '보인' 뒤의 확률이 아니다.
        # 3장 8.1절: P(진짜 0개 | 0개로 보임) = P(N=0) / sum_n P(N=n) * (1-p)^n
        n_vals, cnt = np.unique(reg, return_counts=True)
        p_n = cnt / cnt.sum()
        p_obs0 = float(np.sum(p_n * (1 - p) ** n_vals))
        posterior = prior0 / p_obs0
        # [수정 2] 판정을 시험지의 네 가지로: 기업 컷라인 0.80, 공공 컷라인 0.20
        if posterior >= CUT_BIZ:
            verdict = "공백 판정"
        elif posterior >= CUT_PUB:
            verdict = "현장 확인 대상"
        else:
            verdict = "판정 불가"
    post_txt = "NA" if posterior is None else f"{posterior:.3f}"
    print(f"{name:<6}{int(reg.sum()):>7}{osm_total[k]:>7}{p:>8.3f}{prior0:>12.3f}{post_txt:>10}  {verdict}")
    key(f"p_{name}", float(p))
    key(f"post_{name}", "NA" if posterior is None else float(posterior))
    key(f"verdict_{name}", verdict)
