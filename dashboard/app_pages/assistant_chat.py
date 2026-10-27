import streamlit as st

import assistant as A
import ui

ui.page_header("Ask the assistant", "Ask any question about the project in your own words.")
st.caption(":material/info: This assistant answers from the project's own results and pages, using live numbers. "
           "It is not an AI chatbot: if it does not recognise a question, it suggests related ones instead of guessing. "
           "Answers about the controller use your current simulation settings.")

facts = ui.assistant_facts(ui.cfg_key(ui.scenario()))
history = st.session_state.setdefault("chat_history", [])


def ask(question: str):
    a = A.answer(question, facts)
    history.append({"role": "user", "text": question})
    history.append({"role": "assistant", "text": a.text, "page": a.page, "suggestions": a.suggestions})


def clear():
    history.clear()


if not history:
    with st.chat_message("assistant", avatar=":material/support_agent:"):
        st.markdown("Hi! I can explain the project, the method, the results and every decision. Try one of these:")
        with st.container(horizontal=True):
            for i, q in enumerate(A.SUGGESTED):
                st.button(q, key=f"sugg_{i}", on_click=ask, args=(q,))

for n, msg in enumerate(history):
    with st.chat_message(msg["role"], avatar=":material/support_agent:" if msg["role"] == "assistant" else ":material/person:"):
        st.markdown(msg["text"])
        if msg["role"] == "assistant":
            if msg.get("page"):
                st.page_link(msg["page"][0], label=f"Open: {msg['page'][1]}", icon=":material/arrow_forward:")
            if n == len(history) - 1 and msg.get("suggestions"):
                st.caption("You could also ask:")
                with st.container(horizontal=True):
                    for i, q in enumerate(msg["suggestions"]):
                        st.button(q, key=f"follow_{n}_{i}", on_click=ask, args=(q,))

if history:
    st.button("Clear conversation", icon=":material/delete:", on_click=clear)

if question := st.chat_input("Ask about the project, e.g. 'Why was the move rejected?'"):
    ask(question)
    st.rerun()
