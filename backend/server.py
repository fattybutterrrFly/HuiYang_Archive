"""文学情感共鸣 RAG — Flask API"""
import os
import sys

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from build_index import build_index
from rag_engine import run_pipeline, VectorStore, LLMChat

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)


@app.route("/")
def index():
    return send_from_directory("../frontend", "index.html")


@app.route("/api/health")
def health():
    index_ok = os.path.exists(os.path.join(config.INDEX_DIR, "corpus.jsonl"))
    return jsonify({
        "ok": True,
        "index_built": index_ok,
        "deepseek_key": bool(os.environ.get("DEEPSEEK_API_KEY")),
    })


@app.route("/api/config-keys", methods=["POST"])
def config_keys():
    """允许前端动态设置 DeepSeek API key"""
    body = request.get_json(force=True, silent=True) or {}
    ds = body.get("deepseek_key") or ""
    if ds:
        os.environ["DEEPSEEK_API_KEY"] = ds.strip()
    return jsonify({"ok": True, "deepseek_set": bool(ds)})


@app.route("/api/build-index", methods=["POST"])
def api_build_index():
    try:
        docs, emb = build_index()
        return jsonify({"ok": True, "n_docs": len(docs), "dim": emb.shape[1]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/recommend", methods=["POST"])
def recommend():
    body = request.get_json(force=True, silent=True) or {}
    q = (body.get("query") or "").strip()
    if not q:
        return jsonify({"ok": False, "error": "query 不能为空"}), 400
    if len(q) > 500:
        return jsonify({"ok": False, "error": "query 过长"}), 400

    if body.get("deepseek_key"):
        os.environ["DEEPSEEK_API_KEY"] = body["deepseek_key"].strip()

    t0 = time.time()
    result = run_pipeline(q)
    result["latency_ms"] = int((time.time() - t0) * 1000)
    result["ok"] = True
    return jsonify(result)


if __name__ == "__main__":
    print("=" * 60)
    print("文学情感共鸣 RAG 推荐系统 — MVP")
    print("请确认：")
    print("  1) 已设置  DEEPSEEK_API_KEY  环境变量（或在前端界面输入）")
    print("  2) 首次运行需构建索引（前端按钮可触发，使用本地 embedding）")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
