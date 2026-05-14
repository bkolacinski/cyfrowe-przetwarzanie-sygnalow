import streamlit as st

st.set_page_config(page_title="Cyfrowe Przetwarzanie Sygnałów", layout="wide")

zadanie1_page = st.Page("pages/zadanie1.py", title="Zadanie 1")
zadanie2_page = st.Page("pages/zadanie2.py", title="Zadanie 2")
zadanie3_page = st.Page("pages/zadanie3.py", title="Zadanie 3")

pg = st.navigation([zadanie1_page, zadanie2_page, zadanie3_page])

pg.run()
