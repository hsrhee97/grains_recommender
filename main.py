# main.py
from typing import Dict, Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from rules_core import (
    SURVEY_SCHEMA,
    GRAINS,
    generate_candidates,
    choose_base_dynamic,
    WEIGHTS,
)

app = FastAPI(
    title="Grain Recommender API",
    description="잡곡 추천 엔진 (설문 → 배합 결과)",
    version="0.1.0",
)

# ====== Pydantic 모델 정의 ======

class RecommendRequest(BaseModel):
    # 설문은 자유형 dict로 받되, 키는 한국어 그대로 사용 (예: '취식 목적', '선호식감/맛', '기피', ...)
    survey: Dict[str, Any]

    # 선택: 곡물 수를 바깥에서 강제로 지정하고 싶으면 사용 (없으면 설문 내부 '곡물 수' 사용)
    grains_count: Optional[int] = None


class RecommendResponse(BaseModel):
    mix: Dict[str, int]           # 예: {"백미": 50, "현미": 30, "귀리": 20}
    scores: Dict[str, float]      # 곡물별 점수 (디버깅/설명용)
    base: str                     # 선택된 베이스 곡물
    used_grains_count: int        # 최종 사용된 곡물 수


# ====== 엔드포인트 ======

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """
    설문(survey)을 받아서 잡곡 배합 추천을 반환하는 엔드포인트.
    """
    survey = dict(req.survey)  # 혹시 모를 pydantic 내부 구조 → 일반 dict

    # grains_count를 요청에서 오버라이드하고 싶으면 여기서 설정
    if req.grains_count is not None:
        survey["곡물 수"] = req.grains_count

    # 핵심 로직 호출 (rules_engine.py)
    mix, scores = generate_candidates(survey)

    # 베이스 곡물은 다시 한 번 계산해서 표시 (generate_candidates 내부와 동일 로직)
    base = choose_base_dynamic(survey, scores, WEIGHTS)

    return RecommendResponse(
        mix=mix,
        scores=scores,
        base=base,
        used_grains_count=len(mix),
    )


@app.get("/health")
def health_check():
    """간단한 헬스체크용 엔드포인트."""
    return {"status": "ok"}

from fastapi.responses import HTMLResponse

@app.get("/survey_schema")
def get_survey_schema():
    """
    survey_schema.json + 기피곡물(multiselect)의 실제 곡물 리스트를 합쳐서 내려줌.
    """
    import copy

    schema = copy.deepcopy(SURVEY_SCHEMA)
    grains_options = sorted(GRAINS["곡물"].astype(str).tolist())

    for sec in schema.get("sections", []):
        if sec.get("type") == "multiselect" and sec.get("options_from") == "grains":
            prepend = sec.get("prepend", [])
            # UI에서 바로 쓸 수 있도록 options 필드를 채워준다
            sec["options"] = prepend + grains_options

    return schema

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
      <meta charset="utf-8" />
      <title>잡곡 추천 설문</title>
      <style>
        body { font-family: sans-serif; max-width: 720px; margin: 24px auto; }
        .field { margin-bottom: 16px; }
        label { display: block; font-weight: 600; margin-bottom: 4px; }
      </style>
    </head>
    <body>
      <h1>잡곡 추천 설문</h1>
      <form id="survey-form"></form>

      <button id="submit-btn" type="button">추천 받기</button>

      <h2>추천 결과</h2>
      <pre id="result"></pre>

      <script>
        const formEl = document.getElementById("survey-form");
        const resultEl = document.getElementById("result");
        const submitBtn = document.getElementById("submit-btn");

        // 1) 스키마 불러와서 UI 자동 생성
        async function loadSchemaAndRender() {
          const res = await fetch("/survey_schema");
          const schema = await res.json();
          renderForm(schema.sections || []);
        }

        function renderForm(sections) {
          sections.forEach(sec => {
            const fieldDiv = document.createElement("div");
            fieldDiv.className = "field";

            const label = document.createElement("label");
            label.textContent = sec.name;
            label.htmlFor = sec.name;
            fieldDiv.appendChild(label);

            let input;

            if (sec.type === "dropdown") {
              input = document.createElement("select");
              input.name = sec.name;
              input.id = sec.name;

              (sec.options || []).forEach(opt => {
                const optionEl = document.createElement("option");
                optionEl.value = opt;
                optionEl.textContent = opt;
                if (sec.default && sec.default === opt) {
                  optionEl.selected = true;
                }
                input.appendChild(optionEl);
              });

            } else if (sec.type === "int") {
              input = document.createElement("input");
              input.type = "number";
              input.name = sec.name;
              input.id = sec.name;
              if (sec.min != null) input.min = sec.min;
              if (sec.max != null) input.max = sec.max;
              if (sec.step != null) input.step = sec.step;
              if (sec.default != null) input.value = sec.default;

            } else if (sec.type === "checkbox") {
              input = document.createElement("input");
              input.type = "checkbox";
              input.name = sec.name;
              input.id = sec.name;
              if (sec.default === true) {
                input.checked = true;
              }

            } else if (sec.type === "multiselect") {
              // 여기서는 <select multiple> 로 구현 (기피곡물)
              input = document.createElement("select");
              input.name = sec.name;
              input.id = sec.name;
              input.multiple = true;

              (sec.options || []).forEach(opt => {
                const optionEl = document.createElement("option");
                optionEl.value = opt;
                optionEl.textContent = opt;
                if (Array.isArray(sec.default) && sec.default.includes(opt)) {
                  optionEl.selected = true;
                }
                input.appendChild(optionEl);
              });

              // multiselect 설명 하나 추가
              const hint = document.createElement("div");
              hint.style.fontSize = "12px";
              hint.textContent = "※ Ctrl(또는 ⌘) 키를 누른 상태로 여러 개 선택할 수 있습니다.";
              fieldDiv.appendChild(hint);
            }

            if (input) {
              fieldDiv.appendChild(input);
            }

            formEl.appendChild(fieldDiv);
          });
        }

        // 2) 제출 시 survey 딕셔너리 만들어서 /recommend 호출
        async function onSubmit() {
          const sectionsRes = await fetch("/survey_schema");
          const schema = await sectionsRes.json();
          const sections = schema.sections || [];

          const formData = new FormData(formEl);
          const survey = {};

          sections.forEach(sec => {
            const name = sec.name;
            if (sec.type === "dropdown") {
              survey[name] = formData.get(name) || sec.default || null;

            } else if (sec.type === "int") {
              const raw = formData.get(name);
              survey[name] = raw !== null ? Number(raw) : sec.default || null;

            } else if (sec.type === "checkbox") {
              survey[name] = formData.get(name) === "on";

            } else if (sec.type === "multiselect") {
              const selected = formData.getAll(name); // 여러 값
              survey[name] = selected.length > 0 ? selected : sec.default || [];
            }
          });

          const payload = { survey };

          const resp = await fetch("/recommend", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });

          const data = await resp.json();
          resultEl.textContent = JSON.stringify(data, null, 2);
        }

        submitBtn.addEventListener("click", onSubmit);
        loadSchemaAndRender();
      </script>
    </body>
    </html>
    """
