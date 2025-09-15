"""
에이전트 핵심 모듈
- 컨텍스트/제약 정의, 배합→메트릭/보상, 피처 변환, 의사 선호쌍 데이터셋, 랭커
- 룰 호출은 포함하지 않음(어댑터에서 rules_core.generate_candidates 를 호출)
"""
from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
except Exception:  # scikit-learn 미설치 환경 허용
    LogisticRegression = None
    roc_auc_score = None

# ------------------------------
# 1) 컨텍스트/제약
# ------------------------------
@dataclass
class Context:
    # 선호/목표 가중치(>=0)
    w_gi: float = 1.0       # GI 낮을수록 좋음
    w_fiber: float = 1.0    # 섬유 많을수록 좋음
    w_protein: float = 0.5  # 단백 많을수록 좋음
    w_texture: float = 0.5  # 식감 매칭 중요도
    # 식감 타깃(0~3 범위 가정)
    t_soft: float = 2.0
    t_sticky: float = 1.0
    t_nutty: float = 1.0
    # 제약
    avoid: Tuple[str, ...] = ()
    gluten_free: bool = False
    total: float = 100.0
    # 🔗 설문 전체를 보관(어댑터에서 사용)
    survey: Optional[dict] = None

# ------------------------------
# 2) 비율 유틸/제약 투영
# ------------------------------

def normalize_blend(blend: Dict[str, float], total: float = 100.0) -> Dict[str,float]:
    arr = np.array(list(blend.values()), dtype=float)
    arr = np.maximum(arr, 0.0)
    s = arr.sum()
    if s <= 0:
        n = len(arr)
        arr = np.ones(n) * (total / max(1, n))
    else:
        arr = arr * (total / s)
    return dict(zip(blend.keys(), arr))


def project_constraints(blend: Dict[str, float], grains_df: pd.DataFrame, ctx: Context,
                        per_grain_minmax: Optional[Dict[str, Tuple[float,float]]] = None) -> Dict[str,float]:
    b = {k: float(v) for k,v in blend.items()}
    # 금지곡물 제거
    for g in ctx.avoid:
        if g in b: b[g] = 0.0
    # 글루텐 프리 제거(그레인 테이블에 gluten 컬럼 필요: 0/1)
    if ctx.gluten_free and "gluten" in grains_df.columns:
        gluten_names = set(grains_df.loc[grains_df["gluten"] > 0.5, "곡물" if "곡물" in grains_df.columns else "name"]) \
                       if "곡물" in grains_df.columns else set(grains_df.loc[grains_df["gluten"] > 0.5, "name"])
        for g in gluten_names:
            if g in b: b[g] = 0.0
    # per-grain min/max
    if per_grain_minmax:
        for g,(gmin,gmax) in per_grain_minmax.items():
            if g in b: b[g] = min(max(b[g], gmin), gmax)
    return normalize_blend(b, total=ctx.total)

# ------------------------------
# 3) 배합 → 메트릭/보상/피처
# ------------------------------

def compute_metrics(grains_df: pd.DataFrame, blend: Dict[str,float]) -> Dict[str,float]:
    w = pd.Series(blend, dtype=float) / max(1e-9, sum(blend.values()))
    key = "곡물" if "곡물" in grains_df.columns else "name"
    df = grains_df.set_index(key).reindex(w.index).fillna(0.0)
    gi = float((df.get("GI", df.get("gi", 0)) * w).sum())
    fiber = float((df.get("섬유", df.get("fiber", 0)) * w).sum())
    protein = float((df.get("단백", df.get("protein", 0)) * w).sum())
    soft = float((df.get("부드러움", df.get("soft", 0)) * w).sum())
    sticky = float((df.get("쫀득", df.get("sticky", 0)) * w).sum())
    nutty = float((df.get("고소", df.get("nutty", 0)) * w).sum())
    gluten = float((df.get("글루텐", df.get("gluten", 0)) * w).sum())
    return dict(gi=gi, fiber=fiber, protein=protein, soft=soft, sticky=sticky, nutty=nutty, gluten=gluten)


def texture_similarity(m: Dict[str,float], ctx: Context) -> float:
    v = np.array([m["soft"], m["sticky"], m["nutty"]], float) / 3.0
    t = np.array([ctx.t_soft, ctx.t_sticky, ctx.t_nutty], float) / 3.0
    d = np.linalg.norm(v - t)
    return float(1.0 - min(1.0, d))


def reward(grains_df: pd.DataFrame, blend: Dict[str,float], ctx: Context) -> float:
    m = compute_metrics(grains_df, blend)
    gi_term = - ctx.w_gi * (m["gi"]/3.0)
    fiber_term = ctx.w_fiber * (m["fiber"]/3.0)
    protein_term = ctx.w_protein * (m["protein"]/3.0)
    tex_term = ctx.w_texture * texture_similarity(m, ctx)
    penalty = 0.0
    if ctx.gluten_free and m["gluten"] > 0.0: penalty -= 5.0
    for g in ctx.avoid:
        if g in blend and blend[g] > 0.5: penalty -= 2.0
    return gi_term + fiber_term + protein_term + tex_term + penalty

FEATURE_KEYS = [
    "gi","fiber","protein","soft","sticky","nutty","gluten",
    "t_soft","t_sticky","t_nutty",
    "w_gi","w_fiber","w_protein","w_texture",
]

def features(grains_df: pd.DataFrame, blend: Dict[str,float], ctx: Context) -> np.ndarray:
    m = compute_metrics(grains_df, blend)
    x = [m[k] for k in ["gi","fiber","protein","soft","sticky","nutty","gluten"]]
    x += [ctx.t_soft, ctx.t_sticky, ctx.t_nutty, ctx.w_gi, ctx.w_fiber, ctx.w_protein, ctx.w_texture]
    return np.array(x, float)

# ------------------------------
# 4) 의사 선호쌍 데이터셋
# ------------------------------

def agent_choose_winner(grains_df: pd.DataFrame, ctx: Context, rA: Dict[str,float], rB: Dict[str,float]) -> int:
    sA, sB = reward(grains_df, rA, ctx), reward(grains_df, rB, ctx)
    eps = np.random.normal(0, 0.05)
    return 0 if (sA + eps) >= sB else 1


def make_pairwise_dataset(
    grains_df: pd.DataFrame,
    contexts: List[Context],
    rule_func: Callable[[Context, pd.DataFrame], Dict[str,float]],
    n_pairs_per_ctx: int = 20,
    perturb_fn: Optional[Callable[[Dict[str,float]], Dict[str,float]]] = None,
    rand_scale: float = 8.0,
    per_grain_minmax: Optional[Dict[str,Tuple[float,float]]] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """쌍대학습용 (ϕ(win)-ϕ(lose), y) + (ϕ(lose)-ϕ(win), 0) 대칭 샘플 생성."""
    random.seed(seed); np.random.seed(seed)
    def _perturb(blend: Dict[str,float], scale: float=10.0) -> Dict[str,float]:
        if len(blend) < 2: return blend
        b = blend.copy()
        a, c = random.sample(list(b.keys()), 2)
        delta = (random.random()*2 - 1.0) * scale
        b[a] = max(0.0, b[a] + delta)
        b[c] = max(0.0, b[c] - delta)
        return b

    X, y = [], []
    for ctx in contexts:
        r0 = project_constraints(rule_func(ctx, grains_df), grains_df, ctx, per_grain_minmax)
        for _ in range(n_pairs_per_ctx):
            r1 = project_constraints((perturb_fn or _perturb)(r0, rand_scale), grains_df, ctx, per_grain_minmax)
            r2 = project_constraints((perturb_fn or _perturb)(r0, rand_scale), grains_df, ctx, per_grain_minmax)
            winner = agent_choose_winner(grains_df, ctx, r1, r2)
            Rw, Rl = (r1, r2) if winner == 0 else (r2, r1)
            fw, fl = features(grains_df, Rw, ctx), features(grains_df, Rl, ctx)
            X.append(fw - fl); y.append(1)
            X.append(fl - fw); y.append(0)  # 대칭 표본 추가(학습 안정)
    return np.vstack(X).astype(float), np.array(y, int)

def perturb_neighbour(blend: dict, scale: float = 8.0) -> dict:
    """두 곡물 간 비율을 ±scale%만큼 이동."""
    if len(blend) < 2:
        return blend
    b = blend.copy()
    a, c = random.sample(list(b.keys()), 2)
    delta = (random.random() * 2 - 1.0) * scale
    b[a] = max(0.0, b[a] + delta)
    b[c] = max(0.0, b[c] - delta)
    return b
# ------------------------------
# 5) 랭커(로지스틱 베이스라인)
# ------------------------------
class Ranker:
    def __init__(self):
        if LogisticRegression is None:
            raise RuntimeError("scikit-learn이 필요합니다. pip install scikit-learn")
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if roc_auc_score is None: return float("nan")
        proba = self.model.predict_proba(X)[:,1]
        return float(roc_auc_score(y, proba))

    def pair_score(self, grains_df: pd.DataFrame, ctx: Context, Ra: Dict[str,float], Rb: Dict[str,float]) -> float:
        fa, fb = features(grains_df, Ra, ctx), features(grains_df, Rb, ctx)
        x = (fa - fb).reshape(1, -1)
        if hasattr(self.model, "decision_function"):
            s = float(self.model.decision_function(x)[0])
        else:
            p = float(self.model.predict_proba(x)[0,1]); s = math.log(max(1e-9,p)/max(1e-9,1-p))
        return s

    def infer_best(self, grains_df, ctx, rule_func,
                   n_candidates: int = 8, rand_scale: float = 8.0,
                   per_grain_minmax=None):
        # 1) 룰 배합 R0
        r0 = project_constraints(rule_func(ctx, grains_df), grains_df, ctx, per_grain_minmax)
        # 2) R0 근방 후보 생성(진짜 섭동)
        cands = [r0]
        for _ in range(n_candidates - 1):
            c = perturb_neighbour(r0, scale=rand_scale)
            c = project_constraints(c, grains_df, ctx, per_grain_minmax)
            cands.append(c)
        # 3) 토너먼트 선택
        best = cands[0]
        for c in cands[1:]:
            if self.pair_score(grains_df, ctx, best, c) < 0:
                best = c
        return best
