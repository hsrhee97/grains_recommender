# 0) Setup
# (필요 시 1회만) 위젯 UI 설치
# !pip install ipywidgets

from pathlib import Path
import json, math
import numpy as np
import pandas as pd

# pd.set_option("display.precision", 2)

CONFIG_DIR = Path("config")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# 경로
GRAINS_CSV = CONFIG_DIR / "grains_catalog.csv"
SECTION_WEIGHTS_CSV = CONFIG_DIR / "section_weights.csv"
SURVEY_TO_NUTRIENTS_CSV = CONFIG_DIR / "survey_to_nutrients.csv"
NUTRIENT_TO_GRAINS_CSV = CONFIG_DIR / "nutrient_to_grains.csv"
SURVEY_SCHEMA_JSON = CONFIG_DIR / "survey_schema.json"
RULE_WEIGHTS_JSON = CONFIG_DIR / "rule_weights.json"

# 1) 로더 & 밸리데이션
def load_grains(path=GRAINS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "태그" in df.columns:
        df["태그"] = df["태그"].fillna("").map(lambda s: str(s).split(";") if isinstance(s, str) else [])
    return df

def load_rule_weights(path=RULE_WEIGHTS_JSON) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_section_weights(path=SECTION_WEIGHTS_CSV, fallback_json=None) -> dict:
    if Path(path).exists():
        df = pd.read_csv(path)
        out = {str(r["section"]): float(r["weight"]) for _, r in df.iterrows()}
        return out
    return dict((fallback_json or {}).get("section_weights", {}))

def load_survey_to_nutrients(path=SURVEY_TO_NUTRIENTS_CSV, fallback_json=None) -> dict:
    # 반환 형태: {section: {option: [nut1, nut2, nut3] } }
    if Path(path).exists():
        df = pd.read_csv(path)
        df = df.fillna("")
        out = {}
        for (sec, opt), g in df.groupby(["section","option"]):
            ranks = g.sort_values("rank")
            lst = [x for x in ranks["nutrient"].tolist() if str(x).strip() != ""]
            out.setdefault(sec, {})[opt] = lst
        return out
    return dict((fallback_json or {}).get("survey_to_nutrients", {}))

def load_nutrient_to_grains(path=NUTRIENT_TO_GRAINS_CSV, fallback_json=None) -> dict:
    # 반환 형태: {nutrient: [g1, g2, g3]}
    if Path(path).exists():
        df = pd.read_csv(path)
        out = {}
        for nut, g in df.groupby("nutrient"):
            ranks = g.sort_values("rank")
            out[nut] = ranks["grain"].tolist()
        return out
    return dict((fallback_json or {}).get("nutrient_to_grains", {}))

def norm_section_weights(sw: dict) -> dict:
    if not sw: return {}
    s = sum(sw.values()) or 1.0
    return {k: float(v)/s for k,v in sw.items()}

def load_survey_schema(path=SURVEY_SCHEMA_JSON) -> dict:
    if not Path(path).exists():
        return {"sections": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

SURVEY_SCHEMA = load_survey_schema()

# === (ADD) 스키마 → 값 공간 생성 ===
def build_value_space_from_schema(schema: dict, grains_df: pd.DataFrame):
    """
    설문 스키마(JSON)로부터 각 질문의 가능한 값 리스트를 생성합니다.
    반환: dict[name] = list-of-values
      - dropdown: options 그대로
      - int: min..max..step
      - checkbox: [False, True]
      - multiselect(grains): ["없음"] + 곡물 목록 (※ 전수폭발 방지를 위해 여기서는 '단일 선택'만 샘플링하도록 만듭니다)
    """
    grains_opts = sorted(grains_df["곡물"].astype(str).tolist())
    space = {}

    for sec in schema.get("sections", []):
        t = sec.get("type"); name = sec.get("name")
        if not t or not name: 
            continue

        if t == "dropdown":
            vals = list(sec.get("options", []))
            # none_value 처리: "없음"을 None으로 치환하고 싶다면 아래 줄 활성화
            if "none_value" in sec and sec["none_value"] in vals:
                # 예: 값 공간에 None도 포함시키고 싶으면:
                # vals = [None if v == sec["none_value"] else v for v in vals]
                pass
            space[name] = vals

        elif t == "int":
            vmin, vmax = sec.get("min", 0), sec.get("max", 10)
            step = sec.get("step", 1)
            space[name] = list(range(int(vmin), int(vmax) + 1, int(step)))

        elif t == "checkbox":
            space[name] = [False, True]

        elif t == "multiselect":
            # 옵션이 grains에서 오면: ["없음"] + 곡물 리스트
            if sec.get("options_from") == "grains":
                base_opts = list(sec.get("prepend", [])) + grains_opts
            else:
                base_opts = list(sec.get("options", []))

            # 전수폭발을 막기 위해 샘플러에서는 '없음' 또는 '단일선택'만 허용
            # (노트북의 sample_surveys가 tuple을 list로 바꿔 주므로, 여기서는 값공간을 ["없음"] + [[g]] 형태로 구성해도 됩니다.)
            singletons = [[g] for g in base_opts if g != "없음"]
            space[name] = [["없음"]] + singletons

        # none_value가 있는 dropdown 등에서 None도 후보에 포함하려면 여기서 추가 가능
        # if sec.get("none_value") is not None and None not in space.get(name, []):
        #     space[name] = list(space[name]) + [None]

    # 하위 호환 기본 필드(스키마에 없으면 단일 후보로 추가)
    for k, v in [("체질", ["보통"]), ("장건강", ["보통"]), ("맛", ["담백/중성"])]:
        if k not in space:
            space[k] = v

    return space

# 로드
GRAINS = load_grains()
WEIGHTS = load_rule_weights()
SECTION_WEIGHTS = norm_section_weights(load_section_weights(fallback_json=WEIGHTS))
SURVEY_TO_NUTRIENTS = load_survey_to_nutrients(fallback_json=WEIGHTS)
NUTRIENT_TO_GRAINS = load_nutrient_to_grains(fallback_json=WEIGHTS)
NUTRIENT_RANK_WEIGHTS = WEIGHTS.get("nutrient_rank_weights", [1.0, 2/3, 1/3])
NUTRIENT_GAIN = float(WEIGHTS.get("nutrient_gain", 0.08))

GRAINS.head()

# 2) 스코어링 유틸
def _feature_sum(row, feature_multipliers: dict, base_multiplier: float = 1.0) -> float:
    mult = base_multiplier
    for feat, coef in (feature_multipliers or {}).items():
        mult *= (1.0 + float(coef) * float(row.get(feat, 0)))
    return mult

def _apply_grain_factor(name: str, mapping: dict) -> float:
    if not mapping: return 1.0
    return float(mapping.get(name, 1.0))

def _penalize_if_threshold(value: float, threshold: float, multiplier: float) -> float:
    if threshold is None or multiplier is None: return 1.0
    return float(multiplier) if float(value) >= float(threshold) else 1.0

def _pref_to_texture_fields(survey: dict) -> dict:
    # "선호식감/맛"을 내부 식감/맛으로 보정
    if "선호식감/맛" not in survey: 
        return survey
    prefer = survey.get("선호식감/맛")
    s = dict(survey)
    if prefer == "고슬밥":
        s.setdefault("식감", "부드러움")
        s.setdefault("맛", "담백/중성")
    elif prefer == "찰진밥":
        s.setdefault("식감", "쫀득/단단")
        s.setdefault("맛", "담백/중성")
    else:
        s.setdefault("식감", "중간")
        s.setdefault("맛", "담백/중성")
    return s

def score_purpose(row, survey, W=WEIGHTS):
    key = survey.get("취식 목적") or survey.get("목적") or "맛중심"
    pconf = W["purpose"].get(key, W["purpose"]["맛중심"])
    mult = _feature_sum(row, pconf.get("feature_multipliers", {}), pconf.get("base_multiplier",1.0))
    # grain_multipliers는 1보다 작으면 패널티, 1보다 크면 보너스
    mult *= _apply_grain_factor(row["곡물"], pconf.get("grain_multipliers", {}))
    mult *= _apply_grain_factor(row["곡물"], pconf.get("grain_bonuses", {}))
    return mult

def score_texture_taste(row, survey, W=WEIGHTS):
    tconf = W["texture_taste"]
    mult = 1.0
    sopt = tconf["식감"].get(survey.get("식감","중간"), {})
    mult *= _feature_sum(row, sopt.get("feature_multipliers", {}))
    mult *= _apply_grain_factor(row["곡물"], sopt.get("grain_bonuses", {}))
    mult *= _penalize_if_threshold(row.get("쫀득",0), sopt.get("chewy_penalty_threshold"), sopt.get("chewy_penalty_multiplier"))
    mopt = tconf["맛"].get(survey.get("맛","담백/중성"), {})
    mult *= _feature_sum(row, mopt.get("feature_multipliers", {}))
    mult *= _apply_grain_factor(row["곡물"], mopt.get("grain_bonuses", {}))
    mult *= _apply_grain_factor(row["곡물"], mopt.get("grain_penalties", {}))
    return mult

def score_constitution_gut(row, survey, W=WEIGHTS):
    cg = W["constitution_gut"]
    mult = 1.0
    copt = cg["체질"].get(survey.get("체질","보통"), {})
    mult *= _feature_sum(row, copt.get("feature_multipliers", {}))
    mult *= _penalize_if_threshold(row.get("섬유",0), copt.get("fiber_penalty_at_least"), copt.get("fiber_penalty_multiplier"))
    gopt = cg["장건강"].get(survey.get("장건강","보통"), {})
    soft_bonus = gopt.get("soft_bonus", {})
    if soft_bonus:
        mult *= float(soft_bonus.get("base", 1.0))
        if "부드러움" in soft_bonus:
            mult *= (1.0 + float(soft_bonus["부드러움"]) * float(row.get("부드러움",0)))
    mult *= _feature_sum(row, gopt.get("feature_multipliers", {}))
    mult *= _penalize_if_threshold(row.get("섬유",0), gopt.get("fiber_penalty_at_least"), gopt.get("fiber_penalty_multiplier"))
    return mult

def score_frequency(row, survey, W=WEIGHTS):
    fopt = W["frequency"].get(survey.get("섭취 빈도","주 3-4회"), {})
    mult = 1.0
    mult *= _apply_grain_factor(row["곡물"], fopt.get("grain_bonuses", {}))
    mult *= _apply_grain_factor(row["곡물"], fopt.get("grain_penalties", {}))
    return mult

def allergen_filter(row, survey, W=WEIGHTS):
    if not survey.get("알레르겐 회피(글루텐)", False):
        return 1.0
    if W["allergen"].get("exclude_gluten_grains_if_true", True):
        if row["곡물"] in W["allergen"]["gluten_grains"]:
            return 0.0
    return 1.0

def nutrient_bonus_per_grain(survey, section_weights=SECTION_WEIGHTS,
                             s2n=SURVEY_TO_NUTRIENTS, n2g=NUTRIENT_TO_GRAINS,
                             rank_w=NUTRIENT_RANK_WEIGHTS, gain=NUTRIENT_GAIN):
    # 모든 섹션을 일반화: 설문에 존재 && s2n에 정의된 섹션만 반영
    score = {g: 0.0 for g in GRAINS["곡물"].tolist()}
    for section, ans in survey.items():
        if section not in s2n: 
            continue
        if ans is None or ans == "": 
            continue
        targets = s2n.get(section, {}).get(ans, [])
        if not targets: 
            continue
        sec_w = float(section_weights.get(section, 0.0))
        for idx, nutrient in enumerate(targets[:3]):
            rw = rank_w[idx] if idx < len(rank_w) else 0.0
            grains = n2g.get(nutrient, [])
            for g in grains[:3]:
                score[g] = score.get(g, 0.0) + sec_w * float(rw)
    return {g: (1.0 + gain * v) for g, v in score.items()}

# 3) 베이스/캡/최소치 & 전체 스코어
def choose_base(survey: dict, W=WEIGHTS):
    br = W["base_rules"]
    triggers = br.get("soft_triggers", {})
    soft = False
    for sec, vals in triggers.items():
        if survey.get(sec) in (vals or []):
            soft = True
            break
    avoid = set(survey.get("기피", []) or survey.get("기피곡물", []) or [])
    def _allowed(x):
        if "없음" in avoid: return True
        if x in avoid: return False
        if survey.get("알레르겐 회피(글루텐)", False) and x in W["allergen"]["gluten_grains"]:
            return False
        return True
    cand = br["soft_base"] if soft else br["default_base"]
    if _allowed(cand): return cand
    alt = "현미" if cand == "백미" else "백미"
    if _allowed(alt): return alt
    return br["fallback"]

def base_min_percent(base: str, survey: dict, W=WEIGHTS) -> int:
    br = W["base_rules"]
    if base == "현미" and survey.get("식감") != "부드러움" and survey.get("장건강") != "민감":
        return br["base_min_percent"].get("현미_relaxed", 35)
    return br["base_min_percent"]["default"]

def caps_and_mins(W=WEIGHTS):
    return W.get("caps", {}), W.get("mins", {})

def compute_scores(survey, purpose_intensity=None, texture_intensity=None, W=WEIGHTS):
    survey = _pref_to_texture_fields(survey)
    sw = SECTION_WEIGHTS
    if purpose_intensity is None:
        purpose_intensity = 1.0 + float(sw.get("취식 목적", 0.0))
    if texture_intensity is None:
        texture_intensity = 1.0 + float(sw.get("선호식감/맛", 0.0))

    nut_bonus = nutrient_bonus_per_grain(survey)

    scores = {}
    for _, row in GRAINS.iterrows():
        name = row["곡물"]
        # 기피
        avoid = set(survey.get("기피", []) or survey.get("기피곡물", []) or [])
        if "없음" not in avoid and name in avoid:
            continue
        base = 1.0

        # 알레르겐 필터
        base *= allergen_filter(row, survey, W)
        if base == 0.0:
            continue

        # 섹션 점수
        base *= (score_purpose(row, survey, W) ** purpose_intensity)
        base *= (score_texture_taste(row, survey, W) ** texture_intensity)
        base *= score_constitution_gut(row, survey, W)
        base *= score_frequency(row, survey, W)

        # 영양 보너스
        base *= nut_bonus.get(name, 1.0)

        scores[name] = max(float(base), 1e-6)
    return scores

def select_top_grains(scores: dict, base: str, n: int):
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    names = [k for k,_ in ordered]
    if base in names: names.remove(base)
    return [base] + names[:max(0, n-1)]

def distribute_with_caps(chosen, scores, base, survey, W=WEIGHTS):
    caps, mins = caps_and_mins(W)
    base_min = base_min_percent(base, survey, W)
    base_share = base_min
    remaining = 100 - base_share

    others = [g for g in chosen if g != base]
    if len(others) == 0:
        return {base: 100}

    w = np.array([scores[g] for g in others], float)
    if w.sum() == 0:
        alloc = {g: remaining/len(others) for g in others}
    else:
        w = w / w.sum()
        alloc = {g: float(remaining * w[i]) for i,g in enumerate(others)}
    alloc[base] = float(base_share)

    # 캡 적용 → 초과분 재분배
    pool = 0.0
    for g in list(alloc.keys()):
        cap = caps.get(g, 100)
        if alloc[g] > cap:
            pool += alloc[g] - cap
            alloc[g] = float(cap)
    if pool > 1e-6:
        redis = [g for g in alloc if alloc[g] < caps.get(g, 100)]
        if redis:
            room = np.array([caps.get(g,100)-alloc[g] for g in redis], float)
            room_sum = room.sum()
            if room_sum > 0:
                room = room / room_sum
                for i,g in enumerate(redis):
                    alloc[g] += float(pool * room[i])

    # 최소치 강제
    for g, minv in (mins or {}).items():
        if g in alloc and alloc[g] < minv:
            need = minv - alloc[g]
            alloc[g] = float(minv)
            donors = [x for x in alloc if x != g and alloc[x] > 0]
            if donors:
                weights = np.array([alloc[x] for x in donors], float)
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                    for i,x in enumerate(donors):
                        alloc[x] = max(0.0, alloc[x] - need * weights[i])

    # 합 100 보정 + 정수화
    total = sum(alloc.values())
    if abs(total-100) > 1e-6:
        scale = 100.0 / total
        for g in alloc:
            alloc[g] *= scale

    floats = alloc.copy()
    ints = {g:int(math.floor(v)) for g,v in floats.items()}
    rem = 100 - sum(ints.values())
    rema = sorted([(g, floats[g]-ints[g]) for g in floats], key=lambda x:x[1], reverse=True)
    for i in range(rem):
        ints[rema[i % len(rema)][0]] += 1
    return ints

def generate_candidates(survey: dict):
    # ① 점수 먼저 계산
    pi = float(WEIGHTS["candidate"].get("purpose_intensity", 1.0))
    ti = float(WEIGHTS["candidate"].get("texture_intensity", 1.0))
    scores = compute_scores(survey, pi, ti, WEIGHTS)

    # ② 점수 기반 베이스 선택
    base = choose_base_dynamic(survey, scores, WEIGHTS)

    # ③ 상위 n개 선택 → 캡/최소치 배분
    n = max(WEIGHTS["candidate"]["n_min"],
            min(WEIGHTS["candidate"]["n_max"], int(survey.get("곡물 수", 5))))
    chosen = select_top_grains(scores, base, n)
    mix = distribute_with_caps(chosen, scores, base, survey, WEIGHTS)
    return mix, scores

def choose_base_dynamic(survey: dict, scores: dict, W=WEIGHTS):
    avoid = set(survey.get("기피", []) or survey.get("기피곡물", []) or [])
    allowed = []
    for _, row in GRAINS.iterrows():
        name = row["곡물"]
        tags = row.get("태그", []) or []
        if "base" not in tags:
            continue
        # 기피/글루텐 제약
        if "없음" not in avoid and name in avoid:
            continue
        if survey.get("알레르겐 회피(글루텐)", False) and name in W["allergen"]["gluten_grains"]:
            continue
        allowed.append(name)

    if allowed:
        # 점수 최댓값을 베이스로
        return max(allowed, key=lambda g: scores.get(g, 0.0))

    # 후보가 없으면 기존 규칙으로 폴백
    return choose_base(survey, W)



