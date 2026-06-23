"""文学情感共鸣 RAG 推荐系统 — 核心引擎
四大节点：Intent Analysis → Retrieval → Review & Rerank → Output Generation
基于 LangGraph 思想，用简单状态机实现。
"""
import os
import json
import re
import numpy as np
from openai import OpenAI

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from sentence_transformers import SentenceTransformer

import config


# ---------- 本地 Embedding 模型（懒加载） ----------
_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
    return _embed_model


# ---------- 索引加载 ----------
class VectorStore:
    def __init__(self, index_dir=config.INDEX_DIR):
        self.docs = []
        self.embeddings = None
        self.meta = {}
        self._model = None
        self._load(index_dir)

    def _load(self, index_dir):
        jsonl_path = os.path.join(index_dir, "corpus.jsonl")
        npy_path = os.path.join(index_dir, "embeddings.npy")
        meta_path = os.path.join(index_dir, "meta.json")
        if not (os.path.exists(jsonl_path) and os.path.exists(npy_path)):
            raise FileNotFoundError("未找到索引，请先运行 build_index.py")
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.docs.append(json.loads(line))
        self.embeddings = np.load(npy_path)
        # 已归一化存储，无需再次归一化
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                self.meta = json.load(f)

    def encode(self, text):
        """使用本地 bge-large-zh-v1.5 编码查询"""
        model = get_embed_model()
        # bge 模型建议加指令前缀
        prefixed = "为这个句子生成表示以用于检索: " + text.replace("\n", " ")[:500]
        vec = model.encode(prefixed, normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)

    def search(self, query_vec, top_k=config.TOP_K_RETRIEVE):
        # embeddings 已归一化，直接点积
        scores = self.embeddings.dot(query_vec)
        idx = np.argsort(scores)[-top_k:][::-1]
        return [(self.docs[i], float(scores[i])) for i in idx]


# ---------- LLM 封装 ----------
class LLMChat:
    def __init__(self):
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("请设置 DEEPSEEK_API_KEY 环境变量")
        self.client = OpenAI(api_key=key, base_url=config.DEEPSEEK_BASE_URL)

    def chat(self, sys_prompt, user_msg, temperature=0.7, json_mode=False):
        kwargs = dict(
            model=config.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=temperature,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content


def parse_json_safe(text):
    """尽力把 LLM 返回的内容解析成 JSON"""
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]+\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    # 退而求其次：手工提取字段
    out = {}
    for key in ("emotions", "imageries", "queries", "score", "reason",
                "title", "author", "excerpt", "why", "recommendation"):
        m = re.search(rf'"{key}"\s*:\s*"([^"]*?)"', text)
        if m:
            out[key] = m.group(1)
    # 数组
    for key in ("emotions", "imageries", "queries"):
        m = re.search(rf'"{key}"\s*:\s*\[[^\]]*\]', text)
        if m:
            try:
                out[key] = json.loads("{" + m.group(0) + "}")[key]
            except Exception:
                pass
    return out


# ---------- 节点 F1：意图解析 ----------
SYS_INTENT = """你是一位精通文学审美与情感心理学的读者。
你的任务：把用户感性、隐喻的心境输入，转化为结构化的检索信号。
请严格输出 JSON：
{
  "emotions": ["情感词1", "情感词2", "情感词3"],
  "imageries": ["意象1", "意象2", "意象3"],
  "queries": ["泛化检索词1", "泛化检索词2", "泛化检索词3"],
  "core_mood": "一句凝练的核心心境"
}
要求：
- emotions 至少 2 个，覆盖孤独/怀旧/平静/怅惘/温柔/焦虑等；
- imageries 至少 2 个，必须是可感知的具象意象（如：冬夜的街灯、海边的雨）；
- queries 是「隐喻化的检索短语」，用于向量数据库检索。例如输入"孤独"，可泛化为"空无一人的街道"、"熄灭的街灯"、"冬夜的旷野"。
- 不要解释、不要输出额外文字。
"""


def intent_analysis(llm, user_query, retry_count=0):
    extra = ""
    if retry_count > 0:
        extra = f"\n（第 {retry_count} 次重试：请更加发散、用更广阔的文学隐喻视角重写，避免与之前的 queries 重复。）"
    out = llm.chat(SYS_INTENT, user_query + extra, temperature=0.8, json_mode=True)
    return parse_json_safe(out)


# ---------- 节点 F2：检索 ----------
def retrieve(store, intent):
    """并发执行 3 个泛化查询，汇总去重，取 top10"""
    queries = intent.get("queries") or [intent.get("core_mood", "")]
    if isinstance(queries, str):
        queries = [queries]
    seen = {}
    for q in queries:
        q_vec = store.encode(q)
        for doc, score in store.search(q_vec, top_k=config.TOP_K_RETRIEVE):
            did = doc["doc_id"]
            if did not in seen or score > seen[did][1]:
                seen[did] = (doc, score)
    ranked = sorted(seen.values(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:config.TOP_K_RETRIEVE]]


# ---------- 节点 F3：评审与重排序 ----------
SYS_REVIEW = """你是一位挑剔而公允的文学编辑。
任务：阅读【用户心境】与一段【候选文本】，判断该文本在「意境」和「情感底色」上是否与用户共鸣。
只输出 JSON：
{
  "score": 0 到 100 的整数,
  "reason": "一句话说明评分理由（40 字以内）"
}
评分原则：
- 只看意象/意境/情绪，不看字面关键词；
- 若文本只是包含"焦虑"等词、但毫无文学质感 → 低分；
- 若文本含蓄、通过场景与氛围产生共鸣 → 高分。
"""


def review_and_rerank(llm, user_query, retrieved_docs):
    """逐段打分，应用多样性控制，返回 (top3, max_score)"""
    scored = []
    for doc in retrieved_docs:
        text = doc["text"][:600]
        user_msg = f"【用户心境】\n{user_query}\n\n【候选文本】\n书名：{doc['book_name']}\n作者：{doc['author']}\n片段：\n{text}"
        try:
            out = llm.chat(SYS_REVIEW, user_msg, temperature=0.3, json_mode=True)
            j = parse_json_safe(out)
            score = int(j.get("score", 0))
        except Exception:
            score = 50
        scored.append((doc, score))

    # 多样性控制：同一本书/同一作者最多 1 段
    seen_author = set()
    seen_book = set()
    final = []
    fallback = []
    for doc, score in sorted(scored, key=lambda x: x[1], reverse=True):
        key_a, key_b = doc["author"], doc["book_name"]
        if key_a not in seen_author and key_b not in seen_book:
            final.append((doc, score))
            seen_author.add(key_a)
            seen_book.add(key_b)
        else:
            fallback.append((doc, score - 5))
    final += fallback
    top_k = final[:config.TOP_K_FINAL]
    max_score = max((s for _, s in top_k), default=0)
    return top_k, max_score


# ---------- 节点 F4：生成推荐语 ----------
SYS_OUTPUT = """你是一位温柔、克制、优雅的读书助理。
你不为了卖书，而只是静静递给读者一段能抚平此刻心境的文字。

请为以下 3 段精选文字分别撰写「推荐理由（Why this）」。
要求：
1. 绝不改动原文一字；
2. 推荐语必须一语中的、文学化、避免模板套话；
3. 紧扣「意境如何呼应/抚平用户的心境」，而非机械复述段落内容。
4. 只输出 JSON：
{
  "recommendations": [
    {"book_name": "书名", "author": "作者", "excerpt": "原文片段（完整引用）", "why": "推荐语（80-120字）"}
  ]
}
"""


def generate_recommendations(llm, user_query, top_docs):
    doc_block = []
    for i, (doc, score) in enumerate(top_docs, 1):
        doc_block.append(f"第 {i} 段：\n书名：{doc['book_name']}\n作者：{doc['author']}\n原文：\n{doc['text'][:500]}\n")
    user_msg = f"【用户此刻的心境】\n{user_query}\n\n" + "\n---\n".join(doc_block)
    out = llm.chat(SYS_OUTPUT, user_msg, temperature=0.9, json_mode=True)
    j = parse_json_safe(out)
    recs = j.get("recommendations") if isinstance(j, dict) else None
    # 幻觉防御：把返回的 excerpt 替换为原文
    if recs:
        for i, (orig_doc, _) in enumerate(top_docs):
            if i < len(recs):
                recs[i]["excerpt"] = orig_doc["text"][:600]
                recs[i]["book_name"] = orig_doc["book_name"]
                recs[i]["author"] = orig_doc["author"]
    return recs


# ---------- 主流程 ----------
def check_crisis(user_query):
    for kw in config.SENSITIVE_KEYWORDS:
        if kw in user_query:
            return True
    return False


def run_pipeline(user_query, deepseek_key=None):
    """完整流程：状态机实现（含重试循环）"""
    if deepseek_key:
        os.environ["DEEPSEEK_API_KEY"] = deepseek_key

    result = {
        "user_query": user_query,
        "intent": None,
        "retry_count": 0,
        "max_score": 0,
        "recommendations": None,
        "crisis": False,
        "crisis_message": None,
        "error": None,
    }

    if check_crisis(user_query):
        result["crisis"] = True
        result["crisis_message"] = config.CRISIS_MESSAGE
        return result

    try:
        llm = LLMChat()
        store = VectorStore()
    except Exception as e:
        result["error"] = f"初始化失败：{e}"
        return result

    retry = 0
    final_docs = None
    intent = None
    while retry <= config.MAX_RETRY:
        intent = intent_analysis(llm, user_query, retry_count=retry)
        result["intent"] = intent
        retrieved = retrieve(store, intent)
        top_docs, max_score = review_and_rerank(llm, user_query, retrieved)
        result["retry_count"] = retry
        result["max_score"] = max_score
        if max_score >= config.SCORE_THRESHOLD and len(top_docs) >= 2:
            final_docs = top_docs
            break
        retry += 1

    if final_docs is None:
        # 兜底：取最近一次 top_docs，哪怕分数不高
        final_docs = top_docs if top_docs else []

    if final_docs:
        recs = generate_recommendations(llm, user_query, final_docs)
        result["recommendations"] = recs
    else:
        result["error"] = "未能找到足够的共鸣段落。请尝试更具体的描述。"

    return result


if __name__ == "__main__":
    q = "我最近工作很累，心里空落落的，想找点温柔安静的文字。"
    r = run_pipeline(q)
    print(json.dumps(r, ensure_ascii=False, indent=2))
