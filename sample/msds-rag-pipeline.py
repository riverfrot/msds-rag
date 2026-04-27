"""
MSDS RAG Pipeline (Updated for New KOSHA OpenAPI Spec & All Sections)
======================================================
① KOSHA 공공API → 화학물질ID(chemId) 목록 수집 → 전체 16개 항목 상세 수집 (/getChemDetail01~16)
② 청크 가공 (msdsItemNameKor, itemDetail 매핑) → JSONL 저장
③ CLOVA Embedding API → 벡터화
④ FAISS 인덱스 구축 → 로컬 저장
⑤ RAG 검색 + CLOVA LLM 최종 답변 생성
"""

import asyncio
import json
import os
import time
from pathlib import Path

import httpx
import numpy as np
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

# ── 환경변수 ───────────────────────────────────────────────────────────────
KOSHA_KEY       = os.getenv("KOSHA_KEY", "YOUR_KOSHA_API_KEY")
CLOVA_API_KEY   = os.getenv("CLOVA_API_KEY", "YOUR_CLOVA_API_KEY")

KOSHA_BASE      = "http://msds.kosha.or.kr/openapi/service/msdschem"

SEARCH_KEYWORDS = [
    "에탄올", "아세톤", "헥산", "벤젠", "톨루엔",
    "메탄올", "이소프로판올", "클로로포름", "황산", "질산",
    "암모니아", "염산", "에틸아세테이트", "디클로로메탄", "아세트산",
    "수산화나트륨", "과산화수소", "포름알데히드", "페놀", "자일렌",
]
CLOVA_EMBED_URL = "https://clovastudio.stream.ntruss.com/testapp/v1/api-tools/embedding/clir-sts-dolphin-v1/"
CLOVA_CHAT_URL  = "https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-DASH-002"

# ── 저장 경로 설정 ─────────────────────────────────────────────────────────
DATA_DIR   = Path("msds_data")
CHUNK_FILE = DATA_DIR / "chunks.jsonl"
INDEX_FILE = DATA_DIR / "faiss.index"
META_FILE  = DATA_DIR / "meta.jsonl"

DATA_DIR.mkdir(exist_ok=True)

# ── MSDS 16개 항목 한글명 매핑 ────────────────────────────────────────────
SECTION_NAMES = {
    1:  "화학제품과 회사에 관한 정보", 2:  "유해성·위험성",
    3:  "구성성분의 명칭 및 함유량", 4:  "응급조치요령",
    5:  "폭발·화재시 대처방법", 6:  "누출사고시 대처방법",
    7:  "취급 및 저장방법", 8:  "노출방지 및 개인보호구",
    9:  "물리화학적 특성", 10: "안정성 및 반응성",
    11: "독성에 관한 정보", 12: "환경에 미치는 영향",
    13: "폐기시 주의사항", 14: "운송에 필요한 정보",
    15: "법적 규제현황", 16: "그 밖의 참고사항",
}


# ══════════════════════════════════════════════════════════════════════════
# 1. KOSHA API — 수집 레이어
# ══════════════════════════════════════════════════════════════════════════

async def fetch_api(client: httpx.AsyncClient, url: str, params: dict,
                    retries: int = 3, debug: bool = False) -> dict:
    for attempt in range(retries):
        try:
            r = await client.get(url, params=params, timeout=30)
            r.raise_for_status()

            if debug:
                print(f"\n[DEBUG RAW RESPONSE]\n{r.text[:800]}\n")

            parsed = {}
            try:
                parsed = r.json()
            except ValueError:
                import xmltodict
                parsed = xmltodict.parse(r.text)

            body = (parsed.get("response") or {}).get("body")
            if body and isinstance(body, dict):
                return body

            body = ((parsed.get("OpenAPI") or {}).get("response") or {}).get("body")
            if body and isinstance(body, dict):
                return body

            if "body" in parsed:
                return parsed["body"]
            if "items" in parsed:
                return parsed

            return {}

        except httpx.HTTPStatusError as e:
            print(f"[HTTP ERROR] {e.response.status_code} | {url}")
            return {}
        except Exception as e:
            if attempt == retries - 1:
                print(f"[ERROR] {url} | {e}")
                return {}
            await asyncio.sleep(1.5 * (attempt + 1))
    return {}


async def _fetch_chem_by_keyword(
    client: httpx.AsyncClient, keyword: str, target: int,
    chem_set: set, rows: int = 100, is_first_call: bool = False
) -> list[str]:
    collected = []
    page = 1

    while len(chem_set) + len(collected) < target:
        body = await fetch_api(
            client,
            f"{KOSHA_BASE}/getChemList",
            {
                "serviceKey": KOSHA_KEY,
                "searchWrd":  keyword,
                "searchCnd":  0,
                "numOfRows":  rows,
                "pageNo":     page,
            },
            debug=is_first_call,
        )
        is_first_call = False

        if not body: break

        items     = body.get("items") or {}
        item_list = items.get("item")  or []
        if isinstance(item_list, dict):
            item_list = [item_list]
        if not item_list: break

        for item in item_list:
            if not isinstance(item, dict): continue

            chem_id = (
                item.get("chemId") or item.get("chemid") or
                item.get("casNo") or item.get("CasNo") or ""
            ).strip()

            if chem_id and chem_id not in chem_set:
                collected.append(chem_id)

        total_raw = body.get("totalCount") or body.get("numOfRows") or 0
        total     = int(total_raw) if str(total_raw).isdigit() else "?"
        print(f"    [{keyword}] page {page} | 건수 {len(item_list)} | 누적 {len(chem_set)+len(collected)}/{total}")

        if len(item_list) < rows: break
        page += 1
        await asyncio.sleep(0.3)

    return collected


async def get_chem_list(client: httpx.AsyncClient, target_count: int = 167) -> list[str]:
    chem_set: set[str] = set()
    print(f"\n[1/4] 화학물질 목록(chemId) 수집 중 (목표: {target_count}개)...")

    for i, keyword in enumerate(SEARCH_KEYWORDS):
        if len(chem_set) >= target_count: break

        new_chem = await _fetch_chem_by_keyword(
            client, keyword, target=target_count,
            chem_set=chem_set, is_first_call=(i == 0)
        )
        before = len(chem_set)
        chem_set.update(new_chem)
        print(f"  ✔ '{keyword}': +{len(chem_set)-before}개 추가 → 총 {len(chem_set)}개\n")
        await asyncio.sleep(0.5)

    return list(chem_set)[:target_count]


async def fetch_section_detail(client: httpx.AsyncClient, chem_id: str, section: int) -> dict | None:
    url    = f"{KOSHA_BASE}/getChemDetail{section:02d}"
    params = {"serviceKey": KOSHA_KEY, "chemId": chem_id}
    body   = await fetch_api(client, url, params)

    if not body: return None

    items = body.get("items") or {}
    if not items: return None

    item_list = items.get("item", [])
    if isinstance(item_list, dict):
        item_list = [item_list]

    if not item_list: return None

    flat = {}
    for idx, itm in enumerate(item_list):
        if not isinstance(itm, dict): continue
        name = itm.get("msdsItemNameKor", f"항목_{idx}")
        detail = itm.get("itemDetail", "")

        if detail and detail not in ("null", "None", ""):
            flat[name] = detail

    if not flat:
        return None

    flat["_chem_id"]      = chem_id
    flat["_section"]      = section
    flat["_section_name"] = SECTION_NAMES.get(section, f"항목{section}")
    return flat


def build_chunk_text(record: dict) -> str:
    chem_id  = record.get("_chem_id", "")
    sec      = record.get("_section", "")
    sec_name = record.get("_section_name", "")

    lines = [f"[ChemID {chem_id}] {sec}. {sec_name}"]

    body_fields = {k: v for k, v in record.items() if not k.startswith("_") and v}

    for k, v in body_fields.items():
        clean_v = str(v).replace("<br>", "\n").replace("&nbsp;", " ").strip()
        lines.append(f"- {k}: {clean_v}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# 2. 데이터 수집 메인 루프
# ══════════════════════════════════════════════════════════════════════════

async def collect_data(target_chunks: int = 1000):
    # 16개 섹션을 모두 순회하므로 전체 화학물질 수 계산 수정
    needed_chem = int((target_chunks / 16) * 1.2) + 1

    limits  = httpx.Limits(max_connections=10, max_keepalive_connections=5)
    timeout = httpx.Timeout(30.0)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        chem_list = await get_chem_list(client, target_count=needed_chem)

        print(f"[2/4] 섹션 상세 수집 시작 (전체 1~16번 항목)")
        print(f"      화학물질 {len(chem_list)}개 × 섹션 16개")
        print(f"      = 최대 {len(chem_list)*16}건 API 호출\n")

        chunks      = []
        error_count = 0
        sem = asyncio.Semaphore(5)

        async def fetch_with_sem(c_id, sec):
            nonlocal error_count
            async with sem:
                await asyncio.sleep(0.1)
                result = await fetch_section_detail(client, c_id, sec)
                if result is None: error_count += 1
                return result

        # TARGET_SECTIONS 배열 대신 range(1, 17) 로 1~16번 모두 호출
        tasks = [fetch_with_sem(c_id, sec) for c_id in chem_list for sec in range(1, 17)]
        results = await tqdm_asyncio.gather(*tasks, desc="수집 중", total=len(tasks))

        print(f"\n  수집 결과: 성공 {len(results)-error_count}건 / 오류 {error_count}건")

        with open(CHUNK_FILE, "w", encoding="utf-8") as f:
            for record in results:
                if record is None: continue
                text = build_chunk_text(record)
                if len(text.strip()) < 30: continue

                chunk = {
                    "chem_id":      record["_chem_id"],
                    "section":      record["_section"],
                    "section_name": record["_section_name"],
                    "text":         text,
                    "raw":          {k: v for k, v in record.items() if not k.startswith("_")},
                }
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                chunks.append(chunk)

                if len(chunks) >= target_chunks: break

    print(f"\n  ✅ 청크 저장 완료: {len(chunks)}개 → {CHUNK_FILE}")
    return chunks


# ══════════════════════════════════════════════════════════════════════════
# 3. CLOVA Embedding API → FAISS 인덱스 구축
# ══════════════════════════════════════════════════════════════════════════

def embed_text(text: str, api_key: str) -> list[float] | None:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    text = text[:500]
    try:
        r = httpx.post(CLOVA_EMBED_URL, headers=headers, json={"text": text}, timeout=30)
        r.raise_for_status()
        return r.json()["result"]["embedding"]
    except Exception as e:
        print(f"  [Embed ERROR] {e}")
        return None

def build_faiss_index(chunks: list[dict]):
    try: import faiss
    except ImportError:
        print("faiss-cpu 가 설치되지 않았습니다: pip install faiss-cpu")
        return

    print(f"\n[3/4] 임베딩 + FAISS 인덱스 구축 ({len(chunks)}개 청크)")
    vectors, meta_records, dim = [], [], None

    for i, chunk in enumerate(tqdm_asyncio(chunks, desc="임베딩 중")):
        vec = embed_text(chunk["text"], CLOVA_API_KEY)
        if vec is None: continue
        if dim is None: dim = len(vec)

        vectors.append(vec)
        meta_records.append({
            "idx":          len(meta_records),
            "chem_id":      chunk["chem_id"],
            "section":      chunk["section"],
            "section_name": chunk["section_name"],
            "text":         chunk["text"],
        })
        if (i + 1) % 10 == 0: time.sleep(0.5)

    if not vectors:
        print("  [ERROR] 임베딩 결과가 없습니다.")
        return

    matrix = np.array(vectors, dtype="float32")
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    faiss.write_index(index, str(INDEX_FILE))
    with open(META_FILE, "w", encoding="utf-8") as f:
        for m in meta_records:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"  ✅ FAISS 저장 완료: {INDEX_FILE} (벡터 {index.ntotal}개)")


# ══════════════════════════════════════════════════════════════════════════
# 4. RAG 검색 + CLOVA LLM 답변 생성
# ══════════════════════════════════════════════════════════════════════════

class MsdsRag:
    def __init__(self):
        import faiss
        if not INDEX_FILE.exists() or not META_FILE.exists():
            raise FileNotFoundError("인덱스가 없습니다. build_pipeline() 을 먼저 실행하세요.")

        self.index = faiss.read_index(str(INDEX_FILE))
        self.meta  = [json.loads(line) for line in open(META_FILE, encoding="utf-8")]
        print(f"[RAG] 인덱스 로드 완료: {self.index.ntotal}개 벡터")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        import faiss
        vec = embed_text(query, CLOVA_API_KEY)
        if vec is None: return []

        q = np.array([vec], dtype="float32")
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, top_k)

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0: continue
            m = self.meta[idx].copy()
            m["score"] = float(score)
            results.append(m)
        return results

    def answer(self, question: str, top_k: int = 5) -> str:
        chunks = self.retrieve(question, top_k=top_k)
        if not chunks: return "관련 MSDS 데이터를 찾을 수 없습니다."

        context_parts = []
        for i, c in enumerate(chunks, 1):
            context_parts.append(
                f"[참고자료 {i}] ChemID {c['chem_id']} / "
                f"{c['section']}. {c['section_name']} (유사도: {c['score']:.3f})\n"
                f"{c['text']}"
            )
        context = "\n\n".join(context_parts)

        system_prompt = (
            "당신은 산업안전보건법 전문 MSDS 작성 AI입니다.\n"
            "아래 참고자료(KOSHA 공공데이터 기반)를 활용하여 질문에 답변하세요.\n"
            "- 참고자료에 없는 수치는 '[확인 필요]'로 표기할 것\n"
            "- 영업비밀 성분은 절대 임의 추정하지 말 것\n"
        )
        user_message = f"[참고자료]\n{context}\n\n[질문]\n{question}"
        headers = {"Authorization": f"Bearer {CLOVA_API_KEY}", "Content-Type": "application/json"}

        try:
            r = httpx.post(CLOVA_CHAT_URL, headers=headers, json={
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
                "maxTokens": 2048, "temperature": 0.3, "topP": 0.8,
            }, timeout=60)
            r.raise_for_status()
            return r.json()["result"]["message"]["content"]
        except Exception as e:
            return f"[LLM 오류] {e}"


# ══════════════════════════════════════════════════════════════════════════
# 5. 실행
# ══════════════════════════════════════════════════════════════════════════

async def build_pipeline(target_chunks: int = 1000):
    print("=" * 60 + "\n  MSDS RAG 파이프라인 시작\n" + "=" * 60)
    if CHUNK_FILE.exists():
        chunks = [json.loads(line) for line in open(CHUNK_FILE, encoding="utf-8")]
        print(f"[SKIP] 기존 청크 {len(chunks)}개 로드 완료")
    else:
        chunks = await collect_data(target_chunks=target_chunks)

    if not INDEX_FILE.exists():
        build_faiss_index(chunks)
    else:
        print(f"\n[SKIP] 기존 인덱스 발견: {INDEX_FILE}")
    print("\n" + "=" * 60 + "\n  ✅ 파이프라인 완료! RAG 사용 준비 완료\n" + "=" * 60)


def demo():
    rag = MsdsRag()
    for q in ["에탄올의 인화점과 폭발 범위를 알려줘", "노말헥산 취급 시 개인보호구"]:
        print(f"\n{'='*60}\nQ: {q}\n{'='*60}")
        chunks = rag.retrieve(q, top_k=3)
        for c in chunks:
            print(f"  - ChemID {c['chem_id']} / {c['section_name']} (score={c['score']:.3f})")
        print(f"\n[MSDS 답변]\n{rag.answer(q, top_k=5)}\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv == "demo":
        demo()
    else:
        asyncio.run(build_pipeline(target_chunks=1000))