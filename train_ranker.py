from __future__ import annotations
import random
import numpy as np
import pandas as pd

import rules_core  # 여러분의 기존 룰 모듈
from agent_core import Context, Ranker, project_constraints, compute_metrics
from agent_adapter import map_survey_to_context, make_pairwise_dataset_from_surveys, rule_recommend_from_rules_core

# 1) 스키마에서 랜덤 설문 샘플 만들기(간단 버전)

def sample_surveys(n: int = 50, seed: int = 0) -> list[dict]:
    random.seed(seed)
    # rules_core 내의 SURVEY_SCHEMA, build_value_space_from_schema 를 활용
    space = rules_core.build_value_space_from_schema(rules_core.SURVEY_SCHEMA, rules_core.GRAINS)
    # 과도한 조합을 피하기 위해 각 섹션에서 임의 샘플 1개 선택
    keys = list(space.keys())
    out = []
    for _ in range(n):
        s = {}
        for k in keys:
            vals = space[k]
            s[k] = random.choice(vals)
            if isinstance(s[k], tuple):  # multiselect는 튜플 → 리스트로
                s[k] = list(s[k])
        # 하위 호환 기본값 채우기
        s.setdefault("체질", "보통"); s.setdefault("장건강", "보통"); s.setdefault("맛", "담백/중성")
        s.setdefault("곡물 수", 5)
        out.append(s)
    return out


if __name__ == "__main__":
    # 2) 설문 샘플 생성 → 데이터셋 만들기
    surveys = sample_surveys(n=60, seed=7)
    grains_df, X, y = make_pairwise_dataset_from_surveys(surveys, n_pairs_per_survey=10, rand_scale=8.0)
    print("Dataset:", X.shape, y.shape)

    # 3) 학습
    ranker = Ranker().fit(X, y)
    print("Train AUC:", ranker.score(X, y))

    # 4) 추론 데모
    test_survey = surveys[0]
    ctx = map_survey_to_context(test_survey)
    r0 = rule_recommend_from_rules_core(ctx, grains_df)
    best = ranker.infer_best(grains_df, ctx, rule_recommend_from_rules_core, n_candidates=12, rand_scale=6.0)
    best = project_constraints(best, grains_df, ctx)

    print("
[테스트 설문]")
    print({k:v for k,v in test_survey.items() if v})

    print("
[룰 배합 R0]")
    for k,v in sorted(r0.items(), key=lambda x:-x[1]):
        if v>0.1: print(f"  {k}: {v:.1f}%")

    print("
[에이전트 추천 배합]")
    for k,v in sorted(best.items(), key=lambda x:-x[1]):
        if v>0.1: print(f"  {k}: {v:.1f}%")

    print("
[지표]")
    print(compute_metrics(grains_df, best))