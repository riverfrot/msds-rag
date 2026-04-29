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
| Embedding | `embedding/v2` (1024 dim, role 무시) | `embedding-{query|passage}` alias (4096 dim, 역할별 모델) |
| Rerank | `reranker` (`citedDocuments` → `{index, score}` 어댑팅) | **identity passthrough** — Upstage는 공개 rerank API 미제공, 정규화된 임베딩으로 ANN 정렬이 곧 rerank |
| Chat | `v3/chat-completions/HCX-005` | OpenAI 호환 `chat/completions`, 기본 `solar-pro2` (옵션 `reasoning_effort=low/medium/high`) |
| 인증 | `Authorization: Bearer ${CLOVA_API_KEY}` | `Authorization: Bearer ${UPSTAGE_API_KEY}` |

> 주의 1: CLOVA reranker는 스칼라 점수 대신 인용 목록만 돌려주므로, 클라이언트 단에서 `1.0 - rank * 0.01` 형태의 단조 감소 의사 점수로 변환해 `Retriever`가 기대하는 `{index, score}` 계약을 맞춥니다.
>
> 주의 2: Upstage는 현재 공개 rerank 엔드포인트를 제공하지 않습니다. `solar-embedding-1-large` 가 정규화 벡터(코사인 == 내적)를 출력하므로 Qdrant ANN 결과 자체가 코사인 정렬입니다. 따라서 `UpstageClient.rerank()` 는 외부 호출 0회의 identity passthrough로 구현해 ANN 순서를 보존하고 `top_n`만 적용합니다.

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

provider 별로 컬렉션이 분리되므로 **사용할 provider 마다 한 번씩** 적재합니다.

```bash
# Naver (1024 dim → msds_corpus_naver)
python -m ingest.ingest_jsonl \
    --jsonl ./sample/msds_data/chunks.jsonl \
    --provider naver

# Upstage (4096 dim → msds_corpus_upstage)
python -m ingest.ingest_jsonl \
    --jsonl ./sample/msds_data/chunks.jsonl \
    --provider upstage
```

샘플 코퍼스 1000 청크 기준 실측: Naver ~3분, Upstage ~10분 (RPM 100 제한 + 재시도).

특징:

- 동일한 `(chem_id, section)`은 UUID5(NAMESPACE_URL)로 결정적 ID를 만들어 **idempotent upsert** — 재실행해도 중복 없이 신규 청크만 들어갑니다.
- HTTP 429 / 5xx에는 `Retry-After` 헤더를 우선 존중하고, 없으면 1→2→4→…→30초 지수 백오프 (최대 6회).
- CLOVA 임베딩 입력 한도 보호용으로 청크 텍스트를 `EMBED_CHAR_CAP=500`자에서 잘라 임베딩 (Upstage도 같은 캡 적용 — 청크 의미는 보존되는 길이).

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

### 4.2 라이브 통합 테스트 (실제 API 호출)

```bash
# 둘 다 (키가 있는 provider만 자동 실행)
pytest -m integration

# 하나만
pytest -m "integration and naver"
pytest -m "integration and upstage"
```

- `CLOVA_API_KEY` / `UPSTAGE_API_KEY` 가 비어있거나 placeholder 면 **해당 provider 의 통합 테스트만 자동 skip** (conftest 의 `pytest_collection_modifyitems`가 provider 별 마커로 분리 처리).
- 각 테스트는 **단건 호출**만 수행해 레이트리밋 영향을 최소화 (embed 1회, rerank 1회 — Upstage는 로컬, chat 1회).

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
| `tests/test_naver_client_integration.py` | CLOVA에 **실제 API 단건 호출** | 라이브, `-m "integration and naver"` |
| `tests/test_upstage_client_unit.py` | Upstage embed/rerank(로컬)/chat HTTP 단건 호출, `solar-pro2` 기본값, `reasoning_effort` 전달 | `respx` 로 httpx 모킹 |
| `tests/test_upstage_client_integration.py` | Upstage에 **실제 API 단건 호출** (rerank는 로컬 검증) | 라이브, `-m "integration and upstage"` |
| `tests/test_cli.py` | CLI 인자 파싱·기본/명시 출력 경로·`--no-save`·`--quiet`·헤더 포맷 | `CliRunner` + 파이프라인 스텁 |

### 4.5 현재 결과

```
$ pytest -q
.............................................................................  [100%]
77 passed in ~6s
```

77건 = 단위/모킹 테스트 70건 + 라이브 통합 7건 (Naver 3 + Upstage 4).

---

## 5. 실행 결과 샘플

동일한 입력 — `HW-Cleaner 200` (Ethanol 45%, IPA 30%, 액체) — 의 **MSDS 9번 항목 (물리·화학적 특성)** 을 두 provider 로 각각 생성한 실제 출력입니다. 같은 시스템 프롬프트 + 같은 코퍼스에서 모델 차이만 비교할 수 있습니다.

### 5.1 `--provider naver` (CLOVA HCX-005)

```
[9. 물리·화학적 특성]

가. 외관 (성상/색상): 투명한 무색의 액체

나. 냄새 / 냄새역치: 알코올 특유의 냄새가 있음 (냄새 역치는 자료 없음)

다. pH: 약 6~8 (pH 값의 정확한 수치는 제품별 시험 결과 참조)

라. 녹는점/어는점: 에탄올과 이소프로필알코올 각각의 녹는점은
    각각 -114℃ 및 -89℃ 임

마. 초기 끓는점과 끓는점 범위: 에탄올의 경우 78.37℃,
    이소프로필알코올의 경우 82.3℃ 임

바. 인화점: 에탄올의 경우 약 13℃, 이소프로필알코올의 경우 약 22℃

사. 증기압 / 증기밀도: 데이터는 제품별 시험 결과 참조 필요함

아. 용해도: 물에 잘 녹음

자. 분배계수: n-옥탄올/물 계수는 제품별 시험 결과 참조 필요함

차. 자연발화온도 / 분해온도: 데이터는 제품별 시험 결과 참조 필요함

카. 점도: 점도는 20°C 에서 약 2~4 cSt, 구체적인 수치는
    제품별 시험 결과 참조 필요함
```

### 5.2 `--provider upstage` (Solar Pro 2)

```
가. 외관 (성상/색상): 투명한 무색 액체

나. 냄새 / 냄새역치:
- Ethanol: 특유의 알코올 냄새 (ECHA)
- Isopropyl Alcohol: 특유의 알코올 냄새 (ECHA)
- 제품: 자료 없음

다. pH: Ethanol 7.0 (중성, 20°C, 추정치) /
        IPA 6.0–8.0 (20°C, 추정치) /
        제품: 제품별 시험 결과 참조

라. 녹는점/어는점: Ethanol -114.1°C / IPA -89°C (ECHA)

마. 초기 끓는점과 끓는점 범위:
    Ethanol 78.37°C / IPA 80.37°C (ECHA)
    제품 추정 범위 78.37–80.37°C

바. 인화점: Ethanol 12.8°C / IPA 11.7°C (폐쇄식, ECHA)
            제품: 11.7°C (주성분 기준)

사. 증기압 / 증기밀도:
    Ethanol 5.95 kPa, 1.59 (20°C, HSDB)
    IPA 4.4 kPa, 2.1 (20°C, HSDB)
    제품: 제품별 시험 결과 참조

아. 용해도: 물과 완전히 혼화 (ECHA, 주성분 기준)

자. 분배계수(log Kow): Ethanol -0.32 / IPA 0.06 (ECHA)
    제품: 제품별 시험 결과 참조

차. 자연발화온도: Ethanol 365°C / IPA 460°C (ECHA)
    제품: 자료 없음

카. 점도: Ethanol 1.2 mPa·s (20°C) / IPA 2.4 mPa·s (20°C, 추정치)
    제품: 제품별 시험 결과 참조
```

### 5.3 확인 포인트

- **양식 보존**: 두 출력 모두 가–카 항목이 누락 없이 채워짐.
- **반(反)환각 동작**: 검색 근거에 없는 수치는 둘 다 일관되게 "제품별 시험 결과 참조" / "자료 없음" 으로 대체.
- **양식 완성도 차이**: Solar Pro 2 가 성분별 분리 표기 + 데이터 출처(ECHA/HSDB)를 더 적극적으로 인용. 다만 더 길어지므로 토큰 비용은 더 큼.
- **저장 산출물**: 두 실행 모두 `output/` 아래에 self-describing 헤더가 붙은 마크다운 파일로 보존됨 — 실제 실행 결과 파일은 `output/HW-Cleaner_200_section09_naver_*.md` 와 `output/HW-Cleaner_200_section09_upstage_*.md` 에서 확인 가능.
