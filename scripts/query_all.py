import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chains.rag_chain import generate_answer
from chains.router import route_query
from utils.redis_memory import save_chat, get_chat

session_id = "user_1"  # later make dynamic

while True:
    query = input("\nAsk: ")

    if query.lower() in ["exit", "quit"]:
        break

   
    history = get_chat(session_id)

    print("\n--- Chat History ---\n")
    for msg in history:
        print(f"User: {msg['user']}")
        print(f"Bot: {msg['bot']}\n")

    
    domain = route_query(query)
    print(f"\nDomain: {domain}")

   
    answer = generate_answer(query, domain)

    print("\nAnswer:\n", answer)

   
    save_chat(session_id, query, answer)