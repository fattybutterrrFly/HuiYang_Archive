"""数据预处理：切片、元数据打标、向量索引"""
import os
import re
import json
import glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import config

os.makedirs(config.INDEX_DIR, exist_ok=True)


def read_text_file(path):
    """读取 GBK/UTF-8 编码的文学作品"""
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("gb18030", "utf-8", "utf-16"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("gb18030", errors="replace")


def extract_book_info(path, text):
    """从文件名+文件头部抽取书名、作者"""
    fname = os.path.basename(path).replace(".txt", "")
    # 默认
    book_name = fname
    author = "村上春树"
    # 尝试从文件前 200 字里抓
    head = text[:200]
    lines = [l.strip() for l in head.splitlines() if l.strip()]
    if lines:
        # 第一行常是书名
        book_name = lines[0][:30]
        for line in lines[1:3]:
            if 1 < len(line) < 15 and re.search(r"[著|作者|译|村上|鲁迅|张爱玲|太宰治|普鲁斯特]", line):
                author = re.sub(r"[（(著译）)]", "", line).strip()
                break
    return book_name, author


def clean_text(text):
    text = re.sub(r"\r", "", text)
    # 去掉分隔线
    text = re.sub(r"-{20,}", "", text)
    # 合并多个空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 去掉全角空格前导（常见排版）
    text = re.sub(r"[　 ]+", " ", text)
    return text.strip()


def split_chunks(text, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """按段落+字符数切块，保证上下文连贯性"""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    buf = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(buf) + len(para) < chunk_size:
            buf = (buf + "\n" + para).strip()
        else:
            if buf:
                chunks.append(buf)
            # 单段过长时再切
            if len(para) > chunk_size:
                # 按逗号句号切
                sentences = re.split(r"(?<=[。！？.!?;；])", para)
                sub = ""
                for s in sentences:
                    if len(sub) + len(s) < chunk_size:
                        sub += s
                    else:
                        if sub:
                            chunks.append(sub)
                        sub = s[-chunk_size:] if len(s) > chunk_size else s
                if sub:
                    chunks.append(sub)
                buf = ""
            else:
                buf = para[-overlap:] if len(para) > overlap else para
    if buf:
        chunks.append(buf)
    return chunks


def get_embeddings(texts, model=None, batch_size=32):
    """使用本地 bge-large-zh-v1.5 模型生成向量"""
    if model is None:
        print("→ 正在加载本地 embedding 模型（首次需下载约 1.3GB）...")
        model = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
    # bge-large-zh-v1.5 建议添加指令前缀以提升检索效果
    prefixed = ["为这个句子生成表示以用于检索: " + t.replace("\n", " ")[:500] for t in texts]
    embeddings = model.encode(prefixed, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return np.array(embeddings, dtype=np.float32)


def build_style_keywords(chunks, top_n=60):
    """无监督抽取一些风格/意象关键词（用作 fallback 元数据）"""
    try:
        vec = TfidfVectorizer(
            token_pattern=r"[\u4e00-\u9fa5]{2,}",
            max_features=2000,
            stop_words=["我们", "你们", "他们", "她们", "自己", "一个", "一种", "没有", "不是", "这样", "那样", "什么", "怎么", "这个", "那个", "这里", "那里", "所以", "但是", "因为", "所以", "已经", "还是", "就是", "可以", "可能", "觉得", "知道", "看见", "起来"]
        )
        mat = vec.fit_transform(chunks)
        freqs = np.asarray(mat.sum(axis=0)).flatten()
        idx = np.argsort(freqs)[-top_n:][::-1]
        return [vec.get_feature_names_out()[i] for i in idx]
    except Exception:
        return []


def build_index():
    """主流程：扫描 raw/ ，切块、打标、生成向量索引"""
    files = sorted(glob.glob(os.path.join(config.RAW_DIR, "*.txt")))
    if not files:
        raise RuntimeError(f"raw/ 目录没有 txt 文件: {config.RAW_DIR}")

    all_chunks = []
    docs = []
    doc_id = 0
    for fp in files:
        text = read_text_file(fp)
        book_name, author = extract_book_info(fp, text)
        cleaned = clean_text(text)
        chunks = split_chunks(cleaned)
        print(f"[{book_name}] by {author} → {len(chunks)} 段")
        for chunk in chunks:
            doc = {
                "doc_id": f"doc_{doc_id:04d}",
                "author": author,
                "book_name": book_name,
                "text": chunk,
                "style": "散文/小说",
                "imageries": [],
            }
            docs.append(doc)
            all_chunks.append(chunk)
            doc_id += 1

    # 抽取高频词，作为简易意象元数据
    keywords = build_style_keywords(all_chunks)
    for d in docs:
        hits = [k for k in keywords if k in d["text"]]
        d["imageries"] = hits[:6]
        d["style"] = "小说片段" if any(x in d["book_name"] for x in ["森林", "风吟", "仓房", "舞"]) else "散文"

    # 本地向量索引
    print(f"→ 总共 {len(docs)} 段。开始 Embedding（本地模型）...")
    embeddings = get_embeddings(all_chunks)
    print(f"→ Embedding 完成，shape={embeddings.shape}")

    # 保存
    index_path = os.path.join(config.INDEX_DIR, "corpus.jsonl")
    with open(index_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    vec_path = os.path.join(config.INDEX_DIR, "embeddings.npy")
    np.save(vec_path, embeddings)

    meta = {
        "n_docs": len(docs),
        "embedding_dim": embeddings.shape[1],
        "model": config.LOCAL_EMBEDDING_MODEL,
        "top_keywords": keywords,
    }
    with open(os.path.join(config.INDEX_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✓ 索引已保存到 {config.INDEX_DIR}")
    return docs, embeddings


if __name__ == "__main__":
    build_index()
