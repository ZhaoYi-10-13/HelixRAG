# Copyright 2024
# Directory: yt-rag/app/services/chat.py

"""
Chat service using Alibaba Qwen (DashScope OpenAI-compatible endpoint).
Generates concise answers with citations when context blocks are provided.
"""

import logging
from typing import List, Dict
from openai import OpenAI
from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# 固定使用 DashScope 的 OpenAI 兼容地址
_DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

SYSTEM_PROMPT = (
    "You are a helpful customer support assistant. "
    "Answer strictly using ONLY the provided context blocks. "
    "When you use information from a block, append its id in square brackets, e.g., [chunk_id]. "
    "Always include at least one citation when an answer is given. "
    "If the context does not contain the answer, reply exactly: I don't know. "
    "Avoid markdown headings or special formatting; keep answers concise and professional."
)

def _format_context(context_blocks: List[Dict]) -> str:
    """格式化检索到的上下文块（便于模型引用并输出 [chunk_id]）"""
    parts = []
    for b in context_blocks:
        cid = b.get("chunk_id", "UNKNOWN_ID")
        txt = (b.get("text") or "").strip()
        parts.append(f"[{cid}] {txt}")
    return "\n\n".join(parts) if parts else "N/A"

class ChatService:
    """Qwen chat completion via DashScope"""

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.aliyun_api_key,
            base_url=_DASHSCOPE_BASE,
        )
        self.model = settings.aliyun_chat_model
        self.temperature = settings.temperature

    async def generate_answer(self, query: str, context_blocks: List[Dict]) -> str:
        """生成简洁的客服答案"""
        try:
            context_text = _format_context(context_blocks)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"You will be given context blocks in the form [chunk_id] text. "
                        f"Use them to answer and include [chunk_id] citations.\n\n"
                        f"Context:\n{context_text}\n\nQuestion:\n{query}"
                    ),
                },
            ]

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=512,
            )
            return resp.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Failed to generate answer via Qwen: {e}")
            raise


# Global instance
chat_service = ChatService()