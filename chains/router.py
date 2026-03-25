from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DOMAINS = ["nutrition", "movies", "tech", "general"]


def route_query(query: str, chat_history: list = None) -> str:
    history_text = ""
    if chat_history:
        recent = chat_history[-4:]
        history_text = "\n".join([
            f"User: {msg['user']}\nAssistant: {msg['bot'][:100]}"
            for msg in recent
        ])

    prompt = f"""Classify this query into one domain: nutrition, movies, tech, or general.

nutrition = food, diet, calories, protein, vitamins, recipes
movies = films, actors, directors, bollywood, hollywood, plot, genre
tech = programming, code, errors, APIs, software, frameworks
general = history, economy, science, culture, anything else

{f"Recent conversation:{chr(10)}{history_text}{chr(10)}" if history_text else ""}Query: "{query}"

Reply with ONLY one word: nutrition, movies, tech, or general."""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0,
        )
        domain = response.choices[0].message.content.strip().lower()
        if domain in DOMAINS:
            return domain
    except Exception as e:
        print(f"Router error: {e}")

    # Keyword fallback
    q = query.lower()
    if any(w in q for w in ["food", "protein", "calorie", "diet", "vitamin", "eat", "nutrition", "recipe"]):
        return "nutrition"
    elif any(w in q for w in ["movie", "film", "actor", "director", "bollywood", "hollywood", "plot", "genre"]):
        return "movies"
    elif any(w in q for w in ["code", "error", "api", "python", "javascript", "program", "software", "debug"]):
        return "tech"
    return "general"


def get_domain_emoji(domain: str) -> str:
    return {"nutrition": "🥗", "movies": "🎬", "tech": "💻", "general": "🌐"}.get(domain, "🤖")