"""
Spotify Discovery Engine — AI-Powered Review Analysis
Part 1 of the Graduation Project: Music Discovery Problem Analysis

Data sources: Reddit, App Store, Play Store community discussions
Analysis: Claude AI for thematic coding at scale
"""

import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Discovery Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1DB954 0%, #191414 100%);
        color: white; padding: 2rem 2.5rem; border-radius: 12px; margin-bottom: 2rem;
    }
    .main-header h1 { color: white; margin: 0 0 0.5rem 0; }
    .main-header p  { color: #b3b3b3; margin: 0; }
    .workflow-step {
        background: white; border: 1px solid #e8e8e8; border-radius: 10px;
        padding: 1.2rem 0.8rem; text-align: center; height: 100%;
    }
    .workflow-step .icon { font-size: 2rem; margin-bottom: 0.4rem; }
    .workflow-step b { display: block; font-size: 0.85rem; }
    .workflow-step small { color: #888; font-size: 0.75rem; }
    .theme-card {
        border-left: 4px solid #1DB954; padding: 1rem 1.2rem; margin: 0.5rem 0;
        background: #f8fff8; border-radius: 0 10px 10px 0;
    }
    .quote {
        font-style: italic; color: #555; padding: 0.4rem 1rem;
        border-left: 2px solid #1DB954; margin: 0.3rem 0; background: #fff;
        border-radius: 0 4px 4px 0; font-size: 0.9rem;
    }
    .pill {
        display: inline-block; background: #1DB954; color: white;
        padding: 2px 10px; border-radius: 20px; font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔍 Spotify Discovery Engine</h1>
    <p>AI-powered analysis of why millions of users struggle to discover new music</p>
    <p style="margin-top:0.5rem; font-size:0.8rem;">
        Sources: Reddit (r/spotify, r/SpotifyThrowbacks, r/ifyoulikeblank) · Apple App Store · Play Store Community
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://storage.googleapis.com/pr-newsroom-wp/1/2018/11/Spotify_Logo_RGB_Green.png", width=140)
    st.markdown("---")

    st.subheader("🔑 Live Analysis")
    api_key = st.text_input(
        "Claude API Key (optional)",
        type="password",
        placeholder="sk-ant-api03-...",
        help="Add your Claude API key to run live analysis on freshly fetched data"
    )
    use_live = bool(api_key.strip())

    if use_live:
        st.success("✅ Live analysis enabled")
    else:
        st.info("📊 Showing pre-computed insights\nAdd API key for live analysis")

    st.markdown("---")
    st.subheader("📡 Data Sources")
    fetch_reddit = st.checkbox("Reddit discussions", value=True)
    fetch_appstore = st.checkbox("App Store reviews", value=True)

    if fetch_reddit:
        subreddits = st.multiselect(
            "Subreddits to fetch",
            ["spotify", "SpotifyThrowbacks", "ifyoulikeblank", "musicsuggestions", "indieheads"],
            default=["spotify", "SpotifyThrowbacks"]
        )
        search_queries = st.multiselect(
            "Search queries",
            ["discover new music", "algorithm bubble", "recommendations not working", "tired of same songs"],
            default=["discover new music", "algorithm bubble"]
        )
    else:
        subreddits = []
        search_queries = []

    st.markdown("---")
    st.caption("Graduation Project · July 2026")

# ─── Load precomputed insights ─────────────────────────────────────────────────
@st.cache_data
def load_precomputed():
    p = Path(__file__).parent / "precomputed.json"
    with open(p) as f:
        return json.load(f)

# ─── Live data fetchers ────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_reddit_posts(subreddit, query, limit=25):
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    params = {"q": query, "limit": limit, "sort": "top", "t": "year"}
    headers = {"User-Agent": "DiscoveryEngine/1.0 (research project)"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=12)
        if r.ok:
            posts = []
            for p in r.json()["data"]["children"]:
                d = p["data"]
                posts.append({
                    "title": d["title"],
                    "text": d.get("selftext", "")[:600],
                    "score": d["score"],
                    "subreddit": subreddit,
                    "url": f"https://reddit.com{d['permalink']}"
                })
            return posts
    except Exception:
        pass
    return []

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_appstore_reviews():
    url = "https://itunes.apple.com/us/rss/customerreviews/id=324684580/sortBy=mostRecent/json"
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent": "DiscoveryEngine/1.0"})
        if r.ok:
            entries = r.json().get("feed", {}).get("entry", [])[1:]  # skip first (app metadata)
            return [{
                "title": e.get("title", {}).get("label", ""),
                "text": e.get("content", {}).get("label", ""),
                "rating": e.get("im:rating", {}).get("label", "?"),
                "author": e.get("author", {}).get("name", {}).get("label", "Anonymous")
            } for e in entries[:20]]
    except Exception:
        pass
    return []

# ─── Claude live analysis ──────────────────────────────────────────────────────
def analyze_with_claude(texts: list[str], key: str) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=key)
    combined = "\n---\n".join(t[:400] for t in texts[:30])
    prompt = f"""You are analyzing Spotify user reviews and Reddit discussions about music discovery.

REVIEWS/DISCUSSIONS:
{combined}

Analyze and return ONLY valid JSON with this structure:
{{
  "top_themes": ["theme1 (N mentions)", "theme2", "theme3", "theme4", "theme5"],
  "frustrations": ["frustration1", "frustration2", "frustration3"],
  "unmet_needs": ["need1", "need2", "need3"],
  "notable_quote": "the single most insightful quote from the data"
}}"""
    try:
        msg = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(msg.content[0].text)
    except Exception as e:
        return {"error": str(e)}

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔄 Workflow", "📊 Insights", "💬 Live Data", "👥 User Segments"
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — Workflow
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.subheader("How the Discovery Engine Works")

    cols = st.columns(5)
    steps = [
        ("🕷️", "Data Collection", "Reddit · App Store · Play Store"),
        ("🧹", "Text Cleaning", "Dedup, filter noise, normalize"),
        ("🤖", "Claude Analysis", "Thematic coding at scale via AI"),
        ("🔗", "Triangulation", "Cross-source pattern matching"),
        ("💡", "Insight Output", "Themes · Segments · Needs"),
    ]
    for col, (icon, title, sub) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="workflow-step">
                <div class="icon">{icon}</div>
                <b>{title}</b>
                <small>{sub}</small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    arrow_row = st.columns([1, 0.3, 1, 0.3, 1, 0.3, 1, 0.3, 1])
    for i, col in enumerate(arrow_row):
        if i % 2 == 1:
            col.markdown("<div style='text-align:center; padding-top:0.5rem; font-size:1.5rem; color:#1DB954'>→</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Dataset Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Reddit Posts", "87", "4 subreddits")
    c2.metric("App Store Reviews", "42", "Last 6 months")
    c3.metric("Distinct Themes", "5", "Cross-validated")
    c4.metric("User Segments", "4", "Identified")
    c5.metric("Unmet Needs", "5", "Prioritized")

    st.markdown("---")
    st.subheader("Why This Approach?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Traditional research limitation:** Surveys capture stated preferences, not real frustrations.
Users say "I want more music variety" but don't explain *why* they keep playing the same 5 playlists.

**What reviews/discussions reveal:**
- Unfiltered, unprompted user language
- Real behavior patterns, not idealized responses
- Specific frustration moments (not general sentiment)
- Emerging vocabulary users use to describe the problem
        """)
    with col2:
        st.markdown("""
**Why AI analysis is necessary:**
- 129+ text sources = impossible to manually code at speed
- Claude identifies latent themes across disparate contexts
- Consistent thematic framework applied across all sources
- Quote selection preserves authentic user voice

**Confidence mechanism:**
- Each theme validated across at least 2 independent sources
- Frequency counts reflect cross-platform occurrence
- Representative quotes cherry-picked to maximize specificity
        """)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — Insights
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    insights = load_precomputed()

    st.subheader("Why Users Struggle to Discover New Music")
    st.caption(f"Pre-computed analysis · {insights['sources']['total_analyzed']} sources · {insights['analyzed_at']}")

    # Frequency bar chart
    theme_names = [t["theme"] for t in insights["themes"]]
    theme_freq  = [t["frequency"] for t in insights["themes"]]

    fig = px.bar(
        x=theme_freq, y=theme_names, orientation="h",
        color=theme_freq,
        color_continuous_scale=[[0, "#e8f8ef"], [1, "#1DB954"]],
        labels={"x": "Mention frequency across sources", "y": ""},
        text=theme_freq
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=280, showlegend=False, coloraxis_showscale=False,
        margin=dict(l=0, r=40, t=10, b=0),
        plot_bgcolor="white", paper_bgcolor="white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Theme detail cards
    for theme in insights["themes"]:
        with st.expander(f"**{theme['theme']}** — {theme['frequency']} mentions"):
            st.markdown(theme["description"])
            st.markdown("**Representative quotes from users:**")
            for q in theme["representative_quotes"]:
                st.markdown(f'<div class="quote">"{q}"</div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("😤 Core Frustrations")
        for f in insights["frustrations"]:
            st.markdown(f"▸ {f}")
    with col2:
        st.subheader("✨ Unmet Needs (Opportunities)")
        for n in insights["unmet_needs"]:
            st.markdown(f"▸ {n}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — Live Data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    if not fetch_reddit and not fetch_appstore:
        st.info("Enable data sources in the sidebar to see live data.")
    else:
        all_texts = []

        if fetch_reddit and subreddits:
            st.subheader("🟠 Reddit Discussions")
            all_posts = []

            with st.spinner("Fetching from Reddit..."):
                for sub in subreddits[:2]:
                    for q in search_queries[:2]:
                        posts = fetch_reddit_posts(sub, q)
                        all_posts.extend(posts)
                        time.sleep(0.3)

            if all_posts:
                df = pd.DataFrame(all_posts).drop_duplicates("title")
                df = df.sort_values("score", ascending=False).head(20)
                st.caption(f"Showing top {len(df)} posts by upvote score")

                for _, row in df.iterrows():
                    all_texts.append(row["title"] + " " + row["text"])
                    with st.expander(f"⬆️ {row['score']:,} — {row['title'][:90]}"):
                        if row["text"].strip():
                            st.write(row["text"][:500] + ("..." if len(row["text"]) > 500 else ""))
                        st.caption(f"r/{row['subreddit']}")
                        st.markdown(f"[Open on Reddit ↗]({row['url']})")
            else:
                st.warning("No posts fetched — Reddit may be rate-limiting. Try again in a minute.")

        if fetch_appstore:
            st.subheader("🍎 App Store Reviews")
            with st.spinner("Fetching App Store reviews..."):
                reviews = fetch_appstore_reviews()

            if reviews:
                st.caption(f"Showing {len(reviews)} most recent reviews")
                for rev in reviews:
                    all_texts.append(rev["title"] + " " + rev["text"])
                    stars = "⭐" * int(rev["rating"]) if rev["rating"].isdigit() else rev["rating"]
                    with st.expander(f"{stars} — {rev['title'][:70]}"):
                        st.write(rev["text"][:300])
                        st.caption(f"by {rev['author']}")
            else:
                st.info("App Store reviews unavailable (iTunes RSS may be down). Using pre-computed data.")

        # Live Claude analysis button
        if use_live and all_texts:
            st.markdown("---")
            if st.button("🤖 Run Live Claude Analysis on This Data", type="primary"):
                with st.spinner("Claude is analyzing themes across all fetched content..."):
                    result = analyze_with_claude(all_texts, api_key)

                if "error" in result:
                    st.error(f"Analysis failed: {result['error']}")
                else:
                    st.subheader("Live Analysis Results")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Top Themes**")
                        for t in result.get("top_themes", []):
                            st.markdown(f"• {t}")
                        st.markdown("**Frustrations**")
                        for f in result.get("frustrations", []):
                            st.markdown(f"• {f}")
                    with c2:
                        st.markdown("**Unmet Needs**")
                        for n in result.get("unmet_needs", []):
                            st.markdown(f"• {n}")
                        if "notable_quote" in result:
                            st.markdown("**Standout quote:**")
                            st.markdown(f'<div class="quote">"{result["notable_quote"]}"</div>', unsafe_allow_html=True)
        elif use_live:
            st.info("Enable data sources and fetch data above to run live analysis.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — Segments
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    insights = load_precomputed()

    st.subheader("Who Experiences Discovery Problems?")
    st.caption("Four distinct user segments identified through cross-source analysis")

    colors = ["#1DB954", "#1ed760", "#17a847", "#0d6e30"]
    for i, seg in enumerate(insights["segments"]):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            <div class="theme-card" style="border-color: {colors[i]}">
                <b style="font-size:1.05rem">{seg['name']}</b>
                <p style="margin:0.4rem 0">{seg['description']}</p>
                <small style="color:#555"><b>Core pain:</b> {seg['pain']}</small>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.metric("Estimated share", seg["size_estimate"])

    # Pie chart
    st.markdown("---")
    sizes = [40, 25, 20, 15]
    names = [s["name"] for s in insights["segments"]]
    fig2 = px.pie(
        values=sizes, names=names,
        color_discrete_sequence=["#1DB954", "#1ed760", "#17a847", "#0d6e30"],
        hole=0.4
    )
    fig2.update_layout(
        height=320, margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="v", x=1, y=0.5)
    )
    fig2.update_traces(textinfo="percent+label")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.info("**Focus segment for this project:** The Active Explorers (~25% MAU) — highest intent to discover, most frustrated by current tools, most likely to churn if unserved.")
