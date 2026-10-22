"""
문제 6 (7장). 발주량을 어떻게 정해야 손실이 줄어드는가
========================================================
점포 6곳 x 품목 2종의 일별 수요(store_demand.parquet)에서, 학번으로 점포 하나와 품목 하나를 정한다.
요일별 평균 수요로 예측하고, 예측값을 그대로 발주한다.
결품 1단위 손실 Cu, 잉여 1단위 손실 Co는 7장 8.1절의 가정값이다.

보고서의 문장: "검증용 30%를 무작위로 떼어 평가했으므로 실제 운영에서도 이 손실이 난다."

실행: python exam/q6.py --id 학번
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import PRACTICE, key, parse_id, seed_of

COSTS = {"도시락": (1200, 2000), "우산": (15000, 1500)}   # (Cu, Co)

sid = parse_id()
s = seed_of(sid)
df = pd.read_parquet(PRACTICE / "chapter7" / "data" / "store_demand.parquet")
stores = sorted(df["store_id"].unique())
items = sorted(df["item"].unique())
store, item = stores[s % len(stores)], items[(s // len(stores)) % len(items)]
g = df[(df["store_id"] == store) & (df["item"] == item)].sort_values("day").reset_index(drop=True)
Cu, Co = COSTS[item]
cr = Cu / (Cu + Co)

# [수정 1] 무작위 분할은 미래 날짜를 학습에 섞어 과거를 '예측'하게 만든다. 운영에서는 과거로 미래를
# 예측하므로 시간 순서(day로 정렬됨) 앞 70%를 학습, 뒤 30%를 검증에 쓴다.
n_train = int(len(g) * 0.7)
train, test = g.iloc[:n_train], g.iloc[n_train:]
dow_mean = train.groupby("dow")["demand"].mean()
forecast = test["dow"].map(dow_mean).values
actual = test["demand"].values


def daily_cost(order):
    short = np.maximum(actual - order, 0)
    over = np.maximum(order - actual, 0)
    return float((Cu * short + Co * over).mean())


# 비교용: 예측값을 반올림한 점추정 발주 (원본 방식)
order_point = np.round(forecast)
cost_point = daily_cost(order_point)

# [수정 2] 예측값을 그대로 발주하면 결품(Cu)과 잉여(Co) 손실이 같다고 보는 셈이다(임계비 0.5).
# 7장: 학습 구간 잔차(실제 - 요일 평균 예측)의 임계비 분위수만큼 발주를 보정한다.
resid_train = train["demand"].values - train["dow"].map(dow_mean).values
offset = float(np.quantile(resid_train, cr))
order_nv = np.round(forecast + offset)
cost_newsvendor = daily_cost(order_nv)
saving_pct = (cost_point - cost_newsvendor) / cost_point * 100

print(f"점포 {store}, 품목 {item}, Cu={Cu}, Co={Co}, 임계비={cr:.4f}")
print(f"학습 {len(train)}일 / 검증 {len(test)}일 (시간순 분할: 앞 70% 학습)")
print(f"학습 잔차의 {cr:.4f} 분위수(offset): {offset:+.3f}")
print(f"점추정 발주 일평균 손실: {cost_point:,.0f}원")
print(f"임계비 발주 일평균 손실: {cost_newsvendor:,.0f}원")
print(f"절감률: {saving_pct:.2f}%")
key("critical_ratio", float(cr))
key("cost_per_day", int(round(cost_point)))
key("cost_point", int(round(cost_point)))
key("cost_newsvendor", int(round(cost_newsvendor)))
key("offset", offset)
key("saving_pct", float(saving_pct))
