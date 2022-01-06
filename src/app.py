import streamlit as st
import pandas as pd
import numpy as np 
import sys

import psycopg2 


st.title('postgres connection')



@st.cache(allow_output_mutation=True, hash_funcs={"_thread.RLock": lambda _: None, 'builtins.weakref': lambda _: None})
def init_connection():
    return psycopg2.connect(**st.secrets["postgres"])


conn = init_connection()



@st.cache(ttl=600)
def run_query(query):
	with conn.cursor() as cur:
		cur.execute(query)
		return cur.fetchall()


rows = run_query("SELECT * FROM funds LIMIT 10")

st.write(rows)

for row in rows:
	st.write(f"{row[0]} has a :{row[1]}:")