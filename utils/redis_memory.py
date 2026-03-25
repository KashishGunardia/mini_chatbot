import redis
import json
import uuid
from datetime import datetime, timezone

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True,
)

EXPIRY_SECONDS = 60 * 60 * 24 * 7  # 7 days


def generate_session_id() -> str:
    return str(uuid.uuid4())[:8]


def save_chat(session_id: str, query: str, answer: str, domain: str = "general"):
    key = f"chat:{session_id}"
    message = {
        "user": query,
        "bot": answer,
        "domain": domain,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    redis_client.rpush(key, json.dumps(message))
    redis_client.expire(key, EXPIRY_SECONDS)

    
    title_key = f"title:{session_id}"
    if not redis_client.exists(title_key):
        title = query[:40] + ("..." if len(query) > 40 else "")
        redis_client.set(title_key, title, ex=EXPIRY_SECONDS)


def get_chat(session_id: str, limit: int = 50) -> list:
    key = f"chat:{session_id}"
    messages = redis_client.lrange(key, -limit, -1)
    return [json.loads(msg) for msg in messages]


def clear_chat(session_id: str):
    redis_client.delete(f"chat:{session_id}")
    redis_client.delete(f"title:{session_id}")


def get_all_sessions() -> list:
    """Return all sessions sorted by most recent first, with title and message count."""
    keys = redis_client.keys("chat:*")
    sessions = []
    for key in keys:
        session_id = key.replace("chat:", "")
        messages = redis_client.lrange(key, -1, -1)  # get last message for timestamp
        title = redis_client.get(f"title:{session_id}") or "Untitled session"
        count = redis_client.llen(key)
        last_ts = ""
        if messages:
            try:
                last_msg = json.loads(messages[0])
                last_ts = last_msg.get("timestamp", "")
            except Exception:
                pass
        sessions.append({
            "session_id": session_id,
            "title": title,
            "count": count,
            "last_ts": last_ts,
        })
    # Sort by most recent timestamp
    sessions.sort(key=lambda x: x["last_ts"], reverse=True)
    return sessions