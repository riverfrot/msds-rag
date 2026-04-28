# msds-rag

산업안전보건법 고시 양식에 맞춰 MSDS(물질안전보건자료) 16개 항목 중 **단일 항목**을 RAG로 생성하는 파이프라인.
임베딩·리랭커·LLM은 **Naver Cloud (CLOVA Studio)** 또는 **Upstage(Solar)** 중 하나를 골라 쓸 수 있고, 벡터 저장소는 **Qdrant**.

---

## 1. 설명

기존 MSDS 작성은 16개 항목을 사람이 일일이 자료에서 발췌해 양식에 맞춰 정리해야 했습니다.
이 프로젝트는 **사전 청킹된 MSDS 코퍼스(`chunks.jsonl`)** 를 임베딩해 Qdrant에 적재해두고,
요청 시점에 **CAS 번호 + 항목 번호**를 키로 해당 항목에 필요한 근거만 뽑아 LLM에 전달해
**환각(hallucination)을 최소화한 단일 항목 생성**을 수행합니다.

핵심 원칙:

- **근거 인용 강제**: 검색되지 않은 사실은 "제품별 시험 결과 참조" / "자료 없음"으로 기재 (시스템 프롬프트에서 강제).
- **항목별 양식 준수**: 16개 항목마다 별도의 시스템 프롬프트 (가/나/다 구조, H/P 코드, SI 단위).
- **공급자 추상화**: `model_call(provider, task, ...)` 한 곳에서 임베딩/리랭크/챗을 디스패치 → provider 교체가 한 줄.
- **컬렉션 분리**: 임베딩 차원이 provider별로 다르므로(`naver=1024`, `upstage=4096`) Qdrant 컬렉션을 **물리적으로 분리** (`msds_corpus_naver`, `msds_corpus_upstage`).

---

## 2. 아키텍처

### 2.1 디렉토리 구조

```
msds-scenario/
├── cli/
│   └── msds_cli.py              # `msds-cli` 진입점 (단일 항목 생성)
├── core/
│   ├── clients/
│   │   ├── naver.py             # CLOVA Studio HTTP 클라이언트
│   │   └── upstage.py           # Upstage Solar HTTP 클라이언트
│   ├── model_client.py          # 공급자/태스크 디스패처 + 클라이언트 캐시
│   ├── retriever.py             # Qdrant ANN → provider rerank
│   ├── pipeline.py              # 검색 + 프롬프트 조립 + 챗 호출
│   └── prompts.py               # 16개 항목별 시스템 프롬프트
├── ingest/
│   ├── ingest_jsonl.py          # 사전 청킹된 jsonl을 Qdrant에 적재 (권장)
│   └── ingest_corpus.py         # 원시 .txt 코퍼스를 슬라이딩 윈도우로 적재
├── sample/
│   └── msds_data/chunks.jsonl   # ChemID×Section 단위로 청킹된 코퍼스
├── tests/                       # pytest 스위트 (아래 §4)
├── docker-compose.yml           # Qdrant 단일 컨테이너
└── pyproject.toml
```

### 2.2 데이터 흐름

```
   chunks.jsonl                  ┌─────────────────────────┐
   (chem_id × section)           │ 사용자 요청               │
        │                        │  product / components / │
        ▼                        │  use / form / section   │
 ┌──────────────┐                └──────────┬──────────────┘
 │ ingest_jsonl │                           │
 │  ─ embed     │                           ▼
 │  ─ retry/429 │                ┌──────────────────────┐
 │  ─ upsert    │                │ pipeline.generate_   │
 └──────┬───────┘                │   msds_section()     │
        │                        └──────────┬───────────┘
        ▼                                   │
 ┌──────────────┐         query: product+CAS+항목N
 │   Qdrant     │  ◄────── embed(query, role="query")
 │ msds_corpus_ │  ────►   ANN top_k_first=20 (ScoredPoint)
 │   <provider> │
 └──────────────┘                           │
                                            ▼
                              ┌─────────────────────────────┐
                              │  rerank(query, docs)        │
                              │  → top_k_final=5 evidence   │
                              └────────────┬────────────────┘
                                           │
                                           ▼
                          ┌──────────────────────────────────┐
                          │ chat(system=항목N 프롬프트,        │
                          │      user=제품정보+검색근거)        │
                          └────────────┬─────────────────────┘
                                       │
                                       ▼
                              MSDS 항목 N 본문 (한국어)
```

### 2.3 공급자 추상화

```
core.model_client
    └── model_call(provider, task, **kwargs)
            ├── task="embed"  → client.embed(text, role)
            ├── task="rerank" → client.rerank(query, documents, top_n)
            └── task="chat"   → client.chat(system, user, **chat_kw)
```

| 항목 | Naver (CLOVA) | Upstage (Solar) |
|---|---|---|
| Embedding | `embedding/v2` (1024 dim, role 무시) | `solar-embedding-1-large-{query|passage}` (4096 dim) |
| Rerank | `reranker` (`citedDocuments` → `{index, score}` 어댑팅) | `rerank` (`relevance_score` 그대로) |
| Chat | `v3/chat-completions/HCX-005` | OpenAI 호환 `chat/completions` |

> 주의: CLOVA reranker는 스칼라 점수 대신 인용 목록만 돌려주므로, 클라이언트 단에서 `1.0 - rank * 0.01` 형태의 단조 감소 의사 점수로 변환해 `Retriever`가 기대하는 `{index, score}` 계약을 맞춥니다.

---

## 3. 실행 방법

### 3.1 사전 준비

```bash
# 1) 의존성 설치 (개발용 extras 포함)
pip install -e ".[dev]"

# 2) .env 작성
cp .env.example .env
# .env 안에서 CLOVA_API_KEY 또는 UPSTAGE_API_KEY 설정
#   CLOVA_API_KEY=nv-xxxx...
#   UPSTAGE_API_KEY=up-xxxx...
#   QDRANT_URL=http://localhost:6333
#   MODEL_PROVIDER=naver

# 3) Qdrant 기동 (도커)
docker compose up -d
```

### 3.2 코퍼스 적재

```bash
python -m ingest.ingest_jsonl \
    --jsonl ./sample/msds_data/chunks.jsonl \
    --provider naver
```

특징:

- 동일한 `(chem_id, section)`은 UUID5(NAMESPACE_URL)로 결정적 ID를 만들어 **idempotent upsert** — 재실행해도 중복 없이 신규 청크만 들어갑니다.
- HTTP 429 / 5xx에는 `Retry-After` 헤더를 우선 존중하고, 없으면 1→2→4→…→30초 지수 백오프 (최대 6회).
- CLOVA 임베딩 입력 한도 보호용으로 청크 텍스트를 `EMBED_CHAR_CAP=500`자에서 잘라 임베딩.

### 3.3 단일 항목 생성

```bash
python -m cli.msds_cli \
  --product "HW-Cleaner 200" \
  --components '[
    {"name":"Ethanol","casNumber":"64-17-5","weightPercent":45},
    {"name":"Isopropyl Alcohol","casNumber":"67-63-0","weightPercent":30}
  ]' \
  --use "정밀세정제" \
  --form "액체" \
  --section 9 \
  --provider naver
```

또는 패키지 설치 후 `msds-cli` 콘솔 스크립트로:

```bash
msds-cli --product ... --section 11 ...
```

옵션:

| 옵션 | 필수 | 설명 |
|---|---|---|
| `--product` | ✅ | 제품명 |
| `--components` | ✅ | `name / casNumber / weightPercent` 배열의 JSON 문자열 |
| `--use` | ✅ | 제품 용도 |
| `--form` | ✅ | 물리적 형태 (액체/고체/기체 등) |
| `--section` | ✅ | 1–16 |
| `--provider` |   | `naver` or `upstage`. 기본은 `$MODEL_PROVIDER` (없으면 `naver`) |
| `--output PATH` |   | 결과 저장 경로. 미지정 시 `./output/<제품>_section<NN>_<provider>_<timestamp>.md` |
| `--no-save` |   | 파일 저장 비활성화 (stdout 만) |
| `--quiet` |   | stdout 본문 출력 생략 (저장은 그대로). `[saved] PATH` 안내는 stderr 로 |

저장되는 파일은 자체 설명형(self-describing) 헤더가 붙어 있어 — 제품/공급자/생성 시각/구성 성분/용도/물리적 형태가 본문 위에 함께 적혀 있어 별도 메타데이터 없이도 검토자에게 그대로 전달할 수 있습니다.

```
# MSDS Section 9 — HW-Cleaner 200

- Provider: `naver`
- Generated at: `2026-04-28T21:55:44`
- Use: 정밀세정제
- Physical form: 액체
- Components: `[{"name": "Ethanol", "casNumber": "64-17-5", "weightPercent": 45}, ...]`

---

[9. 물리·화학적 특성]
가. 외관 (성상/색상): 투명한 무색의 액체
...
```

---

## 4. 테스트 코드 실행 방법

### 4.1 전체 (단위 + 모킹)

```bash
pytest -q
```

`pytest.ini_options`에서 `asyncio_mode="auto"` 가 켜져 있어 `async def` 테스트는 별도 데코레이터 없이 동작합니다.

### 4.2 라이브 통합 테스트만 (CLOVA에 실제 호출)

```bash
pytest -m integration
```

- `CLOVA_API_KEY`가 비어있거나 placeholder(`nv-xxxx...`)면 **자동 skip** (conftest의 `pytest_collection_modifyitems` 가 처리).
- 각 테스트는 **단건 호출**만 수행해 레이트리밋 영향을 최소화 (embed 1회, rerank 1회, chat 1회).

### 4.3 통합 빼고 (네트워크 불가 환경)

```bash
pytest -m "not integration"
```

### 4.4 테스트 모듈 구성

| 파일 | 범위 | 호출 방식 |
|---|---|---|
| `tests/test_prompts.py` | 16개 항목 시스템 프롬프트 | 순수 단위 |
| `tests/test_model_client.py` | `model_call` 디스패처, 클라이언트 캐시, `aclose_all` | AsyncMock 스텁 |
| `tests/test_retriever.py` | embed→ANN→rerank 매핑, 빈 결과 처리 | Qdrant/clients 스텁 |
| `tests/test_pipeline.py` | 쿼리 조립, 근거 포맷팅, provider 폴백 | retriever/model_call 스텁 |
| `tests/test_ingest_jsonl.py` | `_embed_with_retry` (429/5xx/Retry-After), `_existing_ids` 페이지네이션 | httpx 에러 직접 주입 |
| `tests/test_naver_client_unit.py` | CLOVA embed/rerank/chat **HTTP 단건 호출** 요청·응답 계약 | `respx` 로 httpx 모킹 |
| `tests/test_naver_client_integration.py` | CLOVA에 **실제 API 단건 호출** | 라이브, `-m integration` |
| `tests/test_cli.py` | CLI 인자 파싱·기본/명시 출력 경로·`--no-save`·`--quiet`·헤더 포맷 | `CliRunner` + 파이프라인 스텁 |

### 4.5 현재 결과

```
$ pytest -q
................................................................         [100%]
64 passed in ~5s
```

---

## 5. 실행 결과 샘플

`HW-Cleaner 200` (Ethanol 45%, IPA 30%, 액체) 의 **MSDS 9번 항목 (물리·화학적 특성)** 을 `--provider naver`로 생성한 실제 출력:

```
[9. 물리·화학적 특성]

가. 외관 (성상/색상): 투명하거나 연한 황색을 띠는 액체

나. 냄새 / 냄새역치: 알코올 특유의 향이 있음 (냄새 역치는 자료 없음)

다. pH: 약 6~8의 중성(pH 값은 제품별 시험 결과 참조)

라. 녹는점/어는점: 에탄올과 이소프로필알코올의 혼합물로 인해
    각각 -114℃ 및 -88℃에서 어는점이 존재함

마. 초기 끓는점과 끓는점 범위: 에탄올의 경우 약 78.37℃,
    이소프로필 알코올의 경우 약 82.3℃
    (끓는점 범위는 두 물질의 혼합으로 인해 변동 가능)

바. 인화점: 약 15℃ ~ 22℃ (인화점은 제품별 시험 결과 참조)

사. 증기압 / 증기밀도: 증기압은 약 44mmHg (20℃),
    증기 밀도는 공기 대비 2.11 g/L
    (증기압 및 밀도는 제품별 시험 결과 참조)

아. 용해도: 물과 완전히 혼합됨; 유기용매에도 잘 용해됨

자. 분배계수(Kow): n-옥탄올/물 분배계수는 각각의 성분에 대해
    특정될 수 있으며, 이는 제품별 시험 결과를 참조해야 함

차. 자연발화온도 / 분해온도: 자연 발화 온도와 분해 온도는
    제품별 시험 결과 참조 필요

카. 점도: 점도는 대략적으로 2~10 cSt (25℃ 기준,
    정확한 수치는 제품별 시험 결과 참조)
```

확인 포인트:

- 양식 (가–카) 가 누락 없이 보존됨.
- 임의의 수치 환각 대신 **"제품별 시험 결과 참조"** 가 일관되게 사용됨 (시스템 프롬프트가 의도한 동작).
- 검색 근거에 등장한 인화점·끓는점은 실제 값이 그대로 인용됨.
