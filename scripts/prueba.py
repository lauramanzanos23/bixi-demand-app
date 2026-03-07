import streamlit as st
import pickle
from pathlib import Path

st.title("🔍 Pickle Test App")

BASE_DIR = Path(__file__).resolve().parent

# Create a tiny test pickle if it doesn't exist
test_pkl = BASE_DIR / "test.pkl"

if not test_pkl.exists():
    obj = {"hello": "world", "number": 42}
    with open(test_pkl, "wb") as f:
        pickle.dump(obj, f)

st.write("📦 Trying to load `test.pkl`...")

try:
    with open(test_pkl, "rb") as f:
        data = pickle.load(f)
    st.success("Loaded OK!")
    st.json(data)
except Exception as e:
    st.error("Failed to load pickle!")
    st.exception(e)
