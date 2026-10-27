import pandas as pd
import streamlit as st

import ui
from knowledge import TERMS

ui.page_header("Glossary", "Plain-English meanings of the terms used in this dashboard.")


df = pd.DataFrame(TERMS, columns=["topic", "term", "meaning"])
topic = st.pills("Topic", ["All"] + list(dict.fromkeys(df.topic)), default="All", key="gl_topic")
q = st.text_input("Search", placeholder="e.g. p95, migration, expert", key="gl_q")
view = df if topic in (None, "All") else df[df.topic == topic]
if q:
    view = view[view.term.str.contains(q, case=False) | view.meaning.str.contains(q, case=False)]
ui.table(view, "Glossary of terms", column_config={"meaning": st.column_config.TextColumn(width="large")},
         height=min(40 + 36 * len(view), 900))
