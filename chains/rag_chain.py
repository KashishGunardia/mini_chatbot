from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

VECTOR_DB_PATH = "vector_db"

GREETINGS = {
    "hi", "hello", "hey", "hii", "helo", "heyy", "yo", "sup",
    "howdy", "greetings", "hi there", "hello there", "hey there"
}


SMALLTALK = {
    "how are you", "how are you doing", "how r u", "how r you",
    "whats up", "what's up", "wyd", "you tell", "you tell me",
    "it was amazing", "it was good", "it was great", "it was fine",
    "same to you", "nice", "okay", "ok", "cool", "great", "awesome",
    "thanks", "thank you", "thx", "ty", "bye", "goodbye", "see you",
    "good morning", "good night", "good evening", "good afternoon",
}


def is_greeting(query: str) -> bool:
    return query.strip().lower().rstrip("!.,") in GREETINGS


def is_smalltalk(query: str) -> bool:
    cleaned = query.strip().lower().rstrip("!.,?")
    return cleaned in SMALLTALK or len(cleaned.split()) <= 4 and not any(
        w in cleaned for w in ["movie", "food", "code", "python", "actor", "recipe", "diet", "film", "tell me about"]
    )


def correct_spelling(query: str) -> str:
    if len(query.strip()) <= 3 or is_greeting(query):
        return query

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a spelling corrector. "
                        "Fix any spelling mistakes or typos in the user's message. "
                        "Keep the meaning and intent exactly the same. "
                        "Do NOT change proper nouns like movie names, actor names, or places unless clearly misspelled. "
                        "Reply with ONLY the corrected sentence — no explanation, no quotes, nothing else."
                    )
                },
                {"role": "user", "content": query}
            ],
            temperature=0,
            max_tokens=100,
        )
        corrected = response.choices[0].message.content.strip()
        if corrected:
            print(f"[Spelling] '{query}' → '{corrected}'")
            return corrected
    except Exception as e:
        print(f"Spelling correction failed: {e}")

    return query


def load_db(domain: str):
    db_path = os.path.join(VECTOR_DB_PATH, domain)
    if not os.path.exists(db_path):
        return None
    return Chroma(
        persist_directory=db_path,
        embedding_function=embedding_model,
    )


def build_messages(query: str, chat_history: list, context: str) -> list:
    system_prompt = """You are a friendly, smart conversational assistant with knowledge about movies, nutrition, technology, and general topics.

Your personality and rules:
- Talk like a helpful, warm friend — natural and conversational, never robotic or formal.
- Always greet back warmly when someone says hi, hello, or hey.
- For casual messages like "it was amazing", "you tell", "how are you" — just respond naturally like a friend would. Do NOT bring up unrelated facts or topics.
- NEVER lose context. Always read the full conversation above before answering.
- Resolve references naturally: "it", "this movie", "that director", "who directed it" all refer to what was last discussed.
- If someone asks a follow-up like "tell me more", "what else", "and?" — continue from exactly where you left off.
- NEVER invent facts, names, or events that weren't mentioned. If unsure, say "I'm not sure about that" honestly.
- Use retrieved knowledge only when it genuinely and clearly helps. If it seems irrelevant, completely ignore it.
- Be concise. No filler. No repeating yourself. Only use lists when the answer genuinely needs them."""

    messages = [{"role": "system", "content": system_prompt}]

    
    if context:
        messages.append({
            "role": "system",
            "content": (
                "Here is some retrieved knowledge that MAY be relevant. "
                "Use it ONLY if it directly answers the question. "
                "If it seems unrelated to what the user is asking, IGNORE it completely:\n"
                + context
            )
        })

    
    for msg in (chat_history or [])[-10:]:
        messages.append({"role": "user", "content": msg["user"]})
        messages.append({"role": "assistant", "content": msg["bot"]})

    messages.append({"role": "user", "content": query})

    return messages


def generate_answer(query: str, domain: str, chat_history: list = None) -> str:
    history = chat_history or []

    
    query = correct_spelling(query)

    
    if is_greeting(query) or is_smalltalk(query):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a warm, friendly conversational assistant. "
                    "Reply naturally and casually to what the user said — like a friend would. "
                    "Keep it short and genuine. "
                    "Do NOT bring up random facts, movie names, or unrelated topics unless the user mentioned them. "
                    "If they say something like 'it was amazing' or 'you tell', respond naturally to that without inventing context."
                )
            }
        ]
        for msg in history[-6:]:
            messages.append({"role": "user", "content": msg["user"]})
            messages.append({"role": "assistant", "content": msg["bot"]})
        messages.append({"role": "user", "content": query})

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.8,
            max_tokens=120,
        )
        return response.choices[0].message.content

   
    context = ""
    db = load_db(domain)
    if db:
        retriever = db.as_retriever(search_kwargs={"k": 5})
        try:
            docs = retriever.invoke(query)
            relevant = [d for d in docs if d.page_content.strip()]
            if relevant:
                context = "\n".join(f"- {d.page_content.strip()}" for d in relevant)
        except Exception as e:
            print(f"Retrieval error: {e}")

    
    messages = build_messages(query, history, context)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )

    return response.choices[0].message.content