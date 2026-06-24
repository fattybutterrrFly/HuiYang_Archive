"""文学情感共鸣 RAG 推荐系统 — 配置"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "raw")
INDEX_DIR = os.path.join(BASE_DIR, "data")

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
TOP_K_RETRIEVE = 10
TOP_K_FINAL = 3
SCORE_THRESHOLD = 60
MAX_RETRY = 2

os.environ.setdefault("DEEPSEEK_API_KEY", "")

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# 本地中文 embedding 模型（bge-large-zh-v1.5）
# 首次运行会自动下载约 1.3GB，之后缓存在本地
LOCAL_EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
EMBEDDING_DIM = 1024

SENSITIVE_KEYWORDS = [
    "自杀", "结束生命", "不想活", "自残", "轻生", "割腕", "跳楼",
    "重度抑郁", "极端痛苦", "无法忍受"
]
CRISIS_MESSAGE = (
    "亲爱的朋友，我读到了你文字里沉重的气息。\n"
    "如果此刻你正身处绝望的边缘，请记住：\n"
    "你并不孤单。请立刻联系以下心理援助热线：\n"
    "• 全国心理援助热线：400-161-9995（24 小时）\n"
    "• 北京心理危机研究与干预中心：010-82951332\n"
    "请给自己一个呼吸。漫漫长夜之后，总会迎来天明。"
)
