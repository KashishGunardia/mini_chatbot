import streamlit as st
from chains.rag_chain import generate_answer
from chains.router import route_query, get_domain_emoji
from utils.redis_memory import save_chat, get_chat, clear_chat, get_all_sessions, generate_session_id

st.set_page_config(page_title="Multi-AI Assistant", page_icon="🤖", layout="centered")

# ── Session state init ────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = generate_session_id()

if "messages" not in st.session_state:
    st.session_state.messages = get_chat(st.session_state.session_id)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Multi-AI Assistant")

    # New session button
    if st.button("➕ New Session", use_container_width=True):
        st.session_state.session_id = generate_session_id()
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # All past sessions
    st.subheader("💬 Past Sessions")
    all_sessions = get_all_sessions()

    if not all_sessions:
        st.caption("No sessions yet. Start chatting!")
    else:
        for s in all_sessions:
            is_active = s["session_id"] == st.session_state.session_id
            label = f"{'▶ ' if is_active else ''}{s['title']}"
            caption = f"{s['count']} messages"

            # Highlight active session
            if is_active:
                st.markdown(f"**{label}**")
                st.caption(f"🟢 Active · {caption}")
            else:
                if st.button(label, key=f"sess_{s['session_id']}", use_container_width=True):
                    st.session_state.session_id = s["session_id"]
                    st.session_state.messages = get_chat(s["session_id"])
                    st.rerun()
                st.caption(caption)

    st.divider()

    # Clear current session
    if st.button("🗑️ Clear Current Session", use_container_width=True):
        clear_chat(st.session_state.session_id)
        st.session_state.messages = []
        st.rerun()

# ── Main chat area ────────────────────────────────────────────────────────────
st.title("🤖 Multi-AI Assistant")
st.caption("Ask me about movies, nutrition, tech, or anything else.")

# Render existing messages
if not st.session_state.messages:
    st.info("Start the conversation by typing a message below.")
else:
    for msg in st.session_state.messages:
        with st.chat_message("user"):
            st.write(msg["user"])
        with st.chat_message("assistant"):
            domain = msg.get("domain", "general")
            st.caption(f"{get_domain_emoji(domain)} {domain.capitalize()}")
            st.write(msg["bot"])

# Chat input
query = st.chat_input("Ask anything...")

if query and query.strip():
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            history = st.session_state.messages
            domain = route_query(query, chat_history=history)
            answer = generate_answer(query, domain, chat_history=history)

        st.caption(f"{get_domain_emoji(domain)} {domain.capitalize()}")
        st.write(answer)

    save_chat(st.session_state.session_id, query, answer, domain)
    st.session_state.messages.append({
        "user": query,
        "bot": answer,
        "domain": domain,
    })