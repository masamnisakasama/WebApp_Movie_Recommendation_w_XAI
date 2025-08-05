import streamlit as st
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from lime_utils import BERTSimilarityExplainer

# TMDbのAPIの設定
TMDB_API_KEY = st.secrets["tmd_api"]["api_key"]
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# BERTのモデル
@st.cache_resource
def load_embed_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_embed_model()

def search_movies(query, max_results=10):
    params = {
        "api_key": TMDB_API_KEY,
        "query": query,
        "language": "ja-JP",
        "include_adult": False,
        "page": 1,
    }
    res = requests.get(f"{TMDB_BASE_URL}/search/movie", params=params)
    return res.json().get("results", [])[:max_results]

def fetch_movie_details(movie_id):
    params = {"api_key": TMDB_API_KEY, "language": "ja-JP"}
    res = requests.get(f"{TMDB_BASE_URL}/movie/{movie_id}", params=params)
    return res.json()

def fetch_similar_movies(movie_id):
    params = {"api_key": TMDB_API_KEY, "language": "ja-JP"}
    res = requests.get(f"{TMDB_BASE_URL}/movie/{movie_id}/similar", params=params)
    return res.json().get("results", [])[:20]

def compute_embeddings(descriptions):
    return model.encode(descriptions, convert_to_numpy=True)

# コサイン類似度による計算
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def rank_similar_movies(selected_id, candidate_movies, candidate_embs, top_k=5):
    selected_desc = fetch_movie_details(selected_id).get("overview", "")
    selected_emb = compute_embeddings([selected_desc])[0]

    scores = [cosine_sim(selected_emb, emb) for emb in candidate_embs]
    sorted_indices = np.argsort(scores)[::-1]

    top_movies = [candidate_movies[i] for i in sorted_indices[:top_k]]
    top_scores = [scores[i] for i in sorted_indices[:top_k]]

    rest_movies = [candidate_movies[i] for i in sorted_indices[top_k:]]
    rest_scores = [scores[i] for i in sorted_indices[top_k:]]

    return (top_movies, top_scores), (rest_movies, rest_scores)

def display_movie(movie, similarity_score=None):
    title = movie.get("title", "タイトル不明")
    year = movie.get("release_date", "????")[:4]
    score = movie.get("vote_average", "N/A")
    genres = ", ".join([g["name"] for g in movie.get("genres", [])]) if "genres" in movie else "不明"
    overview = movie.get("overview", "説明なし")
    poster = movie.get("poster_path")

    with st.expander(f"🎬 {title} ({year})", expanded=False):
        col1, col2 = st.columns([1, 2])
        with col1:
            if poster:
                st.image(f"https://image.tmdb.org/t/p/w300{poster}")
        with col2:
            st.markdown(f"**ジャンル**: {genres}")
            st.markdown(f"**TMDbスコア**: {score}")
            if similarity_score is not None:
                st.markdown(f"**BERT類似度**: `{similarity_score:.3f}`")
            st.write(overview)

# ここからStreamlitのUI部分
st.set_page_config(page_title="映画レコメンダー", layout="wide")
st.markdown("# 🎞️ 類似映画レコメンダー")
st.markdown("###選んだ映画に似ている映画を人工知能がレコメンドします！ 🎥")

query = st.text_input("🔍 映画名を入力してください", value="")

if query:
    results = search_movies(query)
    if not results:
        st.warning("該当する映画が見つかりませんでした。")
    else:
        options = {f"{m['title']} ({m.get('release_date', '')[:4]})": m['id'] for m in results}
        selected_title = st.selectbox("🎯 検索結果から映画を選んでください", list(options.keys()))
        selected_id = options[selected_title]

        selected_info = fetch_movie_details(selected_id)
        genres = ", ".join([g['name'] for g in selected_info.get("genres", [])])
        score = selected_info.get("vote_average", "N/A")
        poster = selected_info.get("poster_path")

        st.markdown("## 🎬 選択された映画の情報")
        col1, col2 = st.columns([1, 2])
        with col1:
            if poster:
                st.image(f"https://image.tmdb.org/t/p/w300{poster}")
        with col2:
            st.write(f"**タイトル**: {selected_title}")
            st.write(f"**ジャンル**: {genres}")
            st.write(f"**TMDbスコア**: {score}")
            st.write(selected_info.get("overview", "説明なし"))

        st.markdown("---")
        st.subheader("🔍 類似映画の推薦中...")

        similar_movies = fetch_similar_movies(selected_id)
        if not similar_movies:
            st.info("類似映画が見つかりませんでした。")
        else:
            descriptions = [m.get("overview", "") for m in similar_movies]
            embeddings = compute_embeddings(descriptions)

            (top5, top5_scores), (others, other_scores) = rank_similar_movies(
                selected_id, similar_movies, embeddings
            )

            st.markdown("## 🌟 BERTによる類似度トップ5")
            for movie, score in zip(top5, top5_scores):
                detail = fetch_movie_details(movie["id"])
                display_movie(detail, score)

                with st.expander("🧠 LIMEでこの映画の類似度の理由を解説します（時間がかかることがあります）"):
                    st.markdown("""
                    **LIMEとは？**  
                    LIMEは「Local Interpretable Model-agnostic Explanations」の略で、  
                    複雑なモデルの予測結果を説明するために、入力テキストのどの単語が結果に影響しているかを教えてくれる技術です。  
                    これにより、なぜこの映画が似ていると判断されたかがわかります。  
                    """)

                    if st.button(f"この映画のLIME解説を表示", key=f"lime_{movie['id']}"):
                        target_desc = detail.get("overview", "")
                        explainer = BERTSimilarityExplainer(selected_info.get("overview", ""))
                        with st.spinner("LIMEで解釈中...しばらくお待ちください"):
                            explanation = explainer.explain(target_desc)
                        for word, weight in explanation.as_list(label=1):
                            color = "green" if weight > 0 else "red"
                            st.markdown(f"- <span style='color:{color}'>**{word}**</span>: `{weight:.3f}`", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("## 🎥 その他の類似映画")
            for movie, score in zip(others, other_scores):
                detail = fetch_movie_details(movie["id"])
                display_movie(detail, score)
