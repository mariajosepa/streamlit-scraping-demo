import streamlit as st
import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load consultant data
@st.cache_data
def load_consultants():
    with open("teamMembers.json") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['text'] = df.apply(lambda row: f"{row['role']} {row['description']}" if pd.notnull(row['description']) else row['role'], axis=1)
    df['embedding'] = model.encode(df['text'].tolist(), show_progress_bar=False).tolist()
    return df

consultant_df = load_consultants()

# Streamlit UI
st.title("🔍 Article-to-Consultant Matcher")

st.markdown("""
Enter an article title and description below. The system will find and rank the consultants most relevant to the content using semantic similarity.
""")

with st.form("article_form"):
    title = st.text_input("Article Title")
    description = st.text_area("Article Description")
    threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.45, 0.01)
    submitted = st.form_submit_button("Find Matches")

if submitted:
    article_text = f"{title} {description}"
    article_embedding = model.encode([article_text])[0].reshape(1, -1)
    consultant_embeddings = np.vstack(consultant_df['embedding'].to_numpy())
    similarities = cosine_similarity(article_embedding, consultant_embeddings)[0]

    consultant_df['similarity_to_article'] = similarities
    matches = consultant_df[consultant_df['similarity_to_article'] >= threshold] \
        .sort_values(by='similarity_to_article', ascending=False)

    st.subheader(f"Top Matches ({len(matches)})")

    # Filter option
    sort_order = st.radio("Sort by Similarity", ["Descending", "Ascending"], horizontal=True)
    matches = matches.sort_values(by='similarity_to_article', ascending=(sort_order == "Ascending"))

    st.dataframe(matches[['name', 'role', 'email', 'similarity_to_article']].reset_index(drop=True), use_container_width=True)
