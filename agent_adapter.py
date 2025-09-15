from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import random
import pandas as pd

from agent_core import Context, project_constraints, features, make_pairwise_dataset, Ranker

# ✅ 여러분의 룰 모듈을 불러옵니다. 파일명이 다르면 아래 import 를 수정하세요.
import rules_core  # (여기에 여러분의 기존 코드 전체가 들어있다고 가정)


# 1) 설문 → Context 간단 매핑(초기가중치; 필요시 조정/외부화)
PURPOSE_TO_WEIGHTS = {
    "혈당관리": dict(w_gi=1.6, w_fiber=1.2, w_protein=0.6, w_texture=0.4),
    "체중 관리": dict(w_gi=1.2, w_fiber=1.3, w_protein=0.7, w_texture=0.5),
    "근육 및 에너지 보강": dict(w_gi=0.8, w_fiber=0.9, w_protein=1.2, w_texture=0.5),
}
TEXTURE_TO_TARGET = {
    "고슬밥": dict(t_soft=2.5, t_sticky=1.0, t_nutty=1.2),
    "찰진밥": dict(t_soft=1.5, t_sticky=2.2, t_nutty=1.0),
}

def map_survey_to_context(survey: dict) -> Context:
    w = PURPOSE_TO_WEIGHTS.get(survey.get("취식 목적"), dict())
    t = TEXTURE_TO_TARGET.get(survey.get("선호식감/맛"), dict())
    avoid = tuple((survey.get("기피", []) or survey.get("기피곡물", []) or []))
    gluten_free = bool(survey.get("알레르겐 회피(글루텐)", False))
    return Context(**w, **t, avoid=avoid, gluten_free=gluten_free, survey=survey)


# 2) 룰 훅: (Context, grains_df) → blend
#    내부적으로 여러분의 rules_core.generate_candidates(survey) 사용

def rule_recommend_from_rules_core(ctx: Context, grains_df: pd.DataFrame) -> Dict[str,float]:
    if ctx.survey is None:
        raise ValueError("Context.survey 가 비어 있습니다. map_survey_to_context를 사용해 주세요.")
    mix, _scores = rules_core.generate_candidates(ctx.survey)
    # rules_core는 이미 제약을 반영한 배합을 내놓지만, 총합/제약을 한 번 더 투영
    return project_constraints({k: float(v) for k,v in mix.items()}, grains_df, ctx)


# 3) 설문 리스트로부터 의사 선호쌍 데이터셋 생성

def make_pairwise_dataset_from_surveys(
    surveys: List[dict],
    n_pairs_per_survey: int = 20,
    rand_scale: float = 8.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, "np.ndarray", "np.ndarray"]:
    """rules_core.GRAINS & generate_candidates 를 사용해 학습셋을 만듭니다."""
    random.seed(seed)
    ctxs = [map_survey_to_context(s) for s in surveys]
    grains_df = rules_core.GRAINS  # 여러분 모듈의 카탈로그
    X, y = make_pairwise_dataset(grains_df, ctxs, rule_recommend_from_rules_core,
                                 n_pairs_per_ctx=n_pairs_per_survey,
                                 rand_scale=rand_scale, seed=seed)
    return grains_df, X, y