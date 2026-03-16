"""
ToxiClean - Professional Streamlit UI
Run: python -m streamlit run app/streamlit_app.py
"""
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd

from modules.pipeline import ToxiCleanPipeline

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ToxiClean — Toxic Speech Neutralizer",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Full CSS Overhaul ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

.stApp { background: #0a0a0f; }
.main .block-container { padding: 2rem 2.5rem 3rem 2.5rem; max-width: 1400px; }

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

[data-testid="stSidebar"] {
    background: #0f0f18 !important;
    border-right: 1px solid #1e1e2e !important;
}
[data-testid="stSidebar"] .block-container { padding: 2rem 1.5rem; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown { color: #c9d1d9 !important; }

[data-testid="stSelectbox"] > div > div,
[data-testid="stSlider"] {
    background: #161625 !important;
    border-color: #2d2d45 !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
}

.stTextArea textarea {
    background: #0d0d1a !important;
    border: 1px solid #2d2d45 !important;
    border-radius: 10px !important;
    color: #e6edf3 !important;
    font-size: 0.95rem !important;
    font-family: 'Inter', sans-serif !important;
    padding: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px #6366f115 !important;
}
.stTextArea textarea::placeholder { color: #4a4a6a !important; }

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.5rem !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    width: 100% !important;
    box-shadow: 0 4px 15px #6366f130 !important;
}
.stButton > button[kind="primary"]:hover { opacity: 0.92 !important; }

.stButton > button[kind="secondary"] {
    background: #161625 !important;
    color: #a0aec0 !important;
    border: 1px solid #2d2d45 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    width: 100% !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #1e1e35 !important;
    color: #e6edf3 !important;
    border-color: #6366f1 !important;
}

.streamlit-expanderHeader {
    background: #0f0f1a !important;
    border: 1px solid #2d2d45 !important;
    border-radius: 8px !important;
    color: #a0aec0 !important;
}
.streamlit-expanderContent {
    background: #0d0d18 !important;
    border: 1px solid #2d2d45 !important;
    border-top: none !important;
}

.stDataFrame { border: 1px solid #2d2d45 !important; border-radius: 10px !important; }

[data-testid="stMetric"] {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 1.6rem !important; }

.stSpinner > div { border-top-color: #6366f1 !important; }
hr { border-color: #1e1e2e !important; margin: 2rem 0 !important; }
.stCaption { color: #4a4a6a !important; font-size: 0.78rem !important; }

/* Custom component styles */
.tc-card-accent-red {
    background: #130a0a; border: 1px solid #3d1515;
    border-left: 4px solid #ef4444;
    border-radius: 0 12px 12px 0;
    padding: 1.25rem 1.5rem; margin: 0.75rem 0;
}
.tc-card-accent-green {
    background: #0a130a; border: 1px solid #153d15;
    border-left: 4px solid #22c55e;
    border-radius: 0 12px 12px 0;
    padding: 1.25rem 1.5rem; margin: 0.75rem 0;
}
.tc-card-accent-blue {
    background: #0a0d18; border: 1px solid #151b3d;
    border-left: 4px solid #6366f1;
    border-radius: 0 12px 12px 0;
    padding: 1.25rem 1.5rem; margin: 0.75rem 0;
}
.tc-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 14px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 700;
    letter-spacing: 0.05em; text-transform: uppercase;
}
.tc-badge-red   { background: #ef444420; color: #ef4444; border: 1px solid #ef444430; }
.tc-badge-green { background: #22c55e20; color: #22c55e; border: 1px solid #22c55e30; }
.tc-badge-blue  { background: #6366f120; color: #818cf8; border: 1px solid #6366f130; }
.tc-pill {
    display: inline-flex; align-items: center; gap: 4px;
    background: #ef444415; color: #fca5a5;
    border: 1px solid #ef444425; border-radius: 20px;
    padding: 3px 10px; font-size: 0.78rem;
    font-weight: 500; margin: 2px;
}
.tc-pill-cat { color: #ef444480; font-size: 0.7rem; }
.tc-section-label {
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: #4a4a6a; margin-bottom: 0.5rem;
}
.tc-neutral-box {
    background: #0a0d18; border: 1px solid #1e2240;
    border-radius: 10px; padding: 1rem 1.25rem;
    font-size: 1rem; color: #93c5fd;
    line-height: 1.6; font-style: italic; margin-top: 0.5rem;
}
.tc-stat {
    display: inline-block; background: #161625;
    border: 1px solid #2d2d45; border-radius: 8px;
    padding: 4px 12px; font-size: 0.78rem;
    color: #6b7280; margin-right: 6px;
}
.tc-stat span { color: #a0aec0; font-weight: 600; }
.tc-step {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 8px 0; border-bottom: 1px solid #1a1a2e;
}
.tc-step-num {
    width: 22px; height: 22px; border-radius: 50%;
    background: #6366f120; border: 1px solid #6366f140;
    color: #818cf8; font-size: 0.7rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.tc-step-text { font-size: 0.82rem; color: #9ca3af; line-height: 1.4; }
.tc-step-text strong { color: #c9d1d9; font-weight: 500; }
.tc-cat { display: flex; align-items: center; gap: 8px; padding: 5px 0; font-size: 0.82rem; color: #9ca3af; }
.tc-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.tc-placeholder {
    border: 1px dashed #2d2d45; border-radius: 12px;
    padding: 3rem 2rem; text-align: center; color: #4a4a6a;
}
.tc-header {
    padding: 2.5rem 0 2rem 0;
    border-bottom: 1px solid #1a1a2e;
    margin-bottom: 2rem;
}
.tc-logo {
    font-size: 2.4rem; font-weight: 800; letter-spacing: -0.03em;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1;
}
.tc-tagline { font-size: 0.9rem; color: #6b7280; margin-top: 0.4rem; }
.tc-version {
    display: inline-block; background: #6366f115;
    border: 1px solid #6366f130; color: #818cf8;
    font-size: 0.7rem; font-weight: 600;
    padding: 3px 10px; border-radius: 20px;
    margin-top: 0.5rem; letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Pipeline ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    return ToxiCleanPipeline()

with st.spinner("Loading ToxiClean engine..."):
    pipeline = load_pipeline()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <div style="font-size:1.3rem;font-weight:800;background:linear-gradient(135deg,#818cf8,#c084fc);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
            🧹 ToxiClean
        </div>
        <div style="font-size:0.75rem;color:#4a4a6a;margin-top:2px;">v1.0 · NLP Toxicity Neutralizer</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="tc-section-label">Configuration</div>', unsafe_allow_html=True)

    strategy = st.selectbox(
        "Neutralization Strategy",
        ["combined", "word_replacement", "rule_based"],
        help="How ToxiClean rewrites toxic text"
    )
    threshold = st.slider(
        "Detection Sensitivity",
        min_value=0.1, max_value=0.9, value=0.3, step=0.05,
        help="Lower = more sensitive"
    )

    st.markdown("<hr style='margin:1.25rem 0;border-color:#1a1a2e;'>", unsafe_allow_html=True)
    st.markdown('<div class="tc-section-label">Pipeline Steps</div>', unsafe_allow_html=True)

    for num, title, desc in [
        ("1","Text Input","Raw user text received"),
        ("2","Preprocessing","Clean & normalize"),
        ("3","Detection","ML toxicity scoring"),
        ("4","Word Analysis","Identify toxic tokens"),
        ("5","Neutralization","Rewrite toxic content"),
        ("6","Output","Clean text delivered"),
    ]:
        st.markdown(f"""
        <div class="tc-step">
            <div class="tc-step-num">{num}</div>
            <div class="tc-step-text"><strong>{title}</strong><br>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='margin:1.25rem 0;border-color:#1a1a2e;'>", unsafe_allow_html=True)
    st.markdown('<div class="tc-section-label">Toxicity Categories</div>', unsafe_allow_html=True)

    for color, name in [
        ("#ef4444","Insult"), ("#f97316","Threat"), ("#eab308","Hate Speech"),
        ("#8b5cf6","Obscene"), ("#ec4899","Aggressive"),
    ]:
        st.markdown(f'<div class="tc-cat"><div class="tc-dot" style="background:{color};"></div>{name}</div>',
                    unsafe_allow_html=True)


# ─── Page Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="tc-header">
  <div style="display:flex;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;gap:1rem;">
    <div>
      <div class="tc-logo">ToxiClean</div>
      <div class="tc-tagline">Toxic Speech Neutralizer · Real-Time NLP Analysis</div>
      <div class="tc-version">NLP · NLTK · scikit-learn · Streamlit</div>
    </div>
    <div style="display:flex;gap:0.75rem;align-items:center;">
      <div style="text-align:center;">
        <div style="font-size:1.4rem;font-weight:700;color:#818cf8;">6</div>
        <div style="font-size:0.7rem;color:#4a4a6a;">Toxicity<br>Labels</div>
      </div>
      <div style="width:1px;height:40px;background:#1e1e2e;"></div>
      <div style="text-align:center;">
        <div style="font-size:1.4rem;font-weight:700;color:#c084fc;">3</div>
        <div style="font-size:0.7rem;color:#4a4a6a;">Neutral<br>Strategies</div>
      </div>
      <div style="width:1px;height:40px;background:#1e1e2e;"></div>
      <div style="text-align:center;">
        <div style="font-size:1.4rem;font-weight:700;color:#818cf8;">91%</div>
        <div style="font-size:0.7rem;color:#4a4a6a;">Model<br>Accuracy</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["  🔍  Single Analysis  ", "  📋  Batch Analysis  "])


# ══ TAB 1 ════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="tc-section-label">Input Text</div>', unsafe_allow_html=True)

        example_options = {
            "— Select a sample —": "",
            "🔴 Insult": "You are such a stupid idiot, completely worthless!",
            "🔴 Threat": "I will hurt you if you don't shut up and go away!",
            "🔴 Obscene": "What the fuck is wrong with you, you dumb asshole!",
            "🔴 Hate speech": "I hate people like you, you're so disgusting and pathetic.",
            "🟢 Assertive (clean)": "I strongly disagree with your approach. Let's discuss.",
            "🟢 Positive": "Have a wonderful day! Your work is truly excellent.",
        }

        selected = st.selectbox("Quick examples", list(example_options.keys()),
                                label_visibility="collapsed")
        default_text = example_options.get(selected, "")

        user_text = st.text_area(
            "Input", value=default_text, height=160,
            placeholder="Paste or type any text here to analyze for toxicity...",
            label_visibility="collapsed"
        )

        c1, c2 = st.columns([3, 1])
        with c1:
            analyze_btn = st.button("Analyze & Neutralize →", type="primary", key="analyze")
        with c2:
            st.button("Clear", type="secondary", key="clear")

        if user_text:
            st.markdown(
                f'<div style="margin-top:0.5rem;">'
                f'<span class="tc-stat"><span>{len(user_text)}</span> chars</span>'
                f'<span class="tc-stat"><span>{len(user_text.split())}</span> words</span>'
                f'</div>', unsafe_allow_html=True
            )

    with col_right:
        st.markdown('<div class="tc-section-label">Analysis Results</div>', unsafe_allow_html=True)

        if analyze_btn and user_text.strip():
            with st.spinner("Analyzing..."):
                pipeline.neutralizer.strategy = strategy
                result = pipeline.analyze(user_text)

            # Verdict
            if result['is_toxic']:
                st.markdown(f"""
                <div class="tc-card-accent-red">
                    <span class="tc-badge tc-badge-red">⬤ Toxic Detected</span>
                    <div style="margin-top:0.75rem;display:flex;gap:0.5rem;flex-wrap:wrap;">
                        <span class="tc-stat">Confidence <span>{result['confidence']:.0%}</span></span>
                        <span class="tc-stat">Intensity <span>{result['intensity_score']:.0%}</span></span>
                        <span class="tc-stat">Toxic words <span>{result['toxic_count']}</span></span>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="tc-card-accent-green">
                    <span class="tc-badge tc-badge-green">⬤ Text Is Clean</span>
                    <div style="margin-top:0.5rem;font-size:0.85rem;color:#6b7280;">
                        No toxic content detected. Safe to publish.
                    </div>
                </div>""", unsafe_allow_html=True)

            # Toxic words
            if result['toxic_words']:
                st.markdown('<div class="tc-section-label" style="margin-top:1.25rem;">Toxic Words Identified</div>',
                            unsafe_allow_html=True)
                pills = " ".join([
                    f'<span class="tc-pill">{w["word"]} <span class="tc-pill-cat">· {w["category"]}</span></span>'
                    for w in result['toxic_words']
                ])
                st.markdown(f'<div style="line-height:2;">{pills}</div>', unsafe_allow_html=True)

            # Category tags
            if result['toxicity_types']:
                st.markdown('<div class="tc-section-label" style="margin-top:1.25rem;">Toxicity Types</div>',
                            unsafe_allow_html=True)
                cat_colors = {
                    'insult':'#ef4444','threat':'#f97316','hate':'#eab308',
                    'obscene':'#8b5cf6','aggressive':'#ec4899',
                    'severe_toxic':'#dc2626','identity_hate':'#b45309'
                }
                tags = " ".join([
                    f'<span style="display:inline-block;background:{cat_colors.get(t,"#6366f1")}18;'
                    f'color:{cat_colors.get(t,"#818cf8")};border:1px solid {cat_colors.get(t,"#6366f1")}30;'
                    f'border-radius:20px;padding:3px 12px;font-size:0.78rem;font-weight:600;margin:2px;">{t}</span>'
                    for t in result['toxicity_types']
                ])
                st.markdown(tags, unsafe_allow_html=True)

            # Neutralized output
            if result['is_toxic']:
                st.markdown('<div class="tc-section-label" style="margin-top:1.25rem;">Neutralized Version</div>',
                            unsafe_allow_html=True)
                st.markdown(f'<div class="tc-neutral-box">"{result["neutral_text"]}"</div>',
                            unsafe_allow_html=True)

                if result['changes_made']:
                    with st.expander("View changes made"):
                        for old, new in result['changes_made']:
                            if isinstance(old, str) and isinstance(new, str) and len(old) < 60:
                                st.markdown(
                                    f'<div style="padding:4px 0;font-size:0.83rem;color:#9ca3af;">'
                                    f'<code style="background:#1a1a2e;color:#fca5a5;padding:2px 6px;border-radius:4px;">{old}</code>'
                                    f' <span style="color:#4a4a6a;">→</span> '
                                    f'<code style="background:#1a1a2e;color:#86efac;padding:2px 6px;border-radius:4px;">{new}</code>'
                                    f'</div>', unsafe_allow_html=True
                                )
            else:
                st.markdown("""
                <div class="tc-neutral-box" style="color:#86efac;border-color:#15803d30;">
                    ✓ Your text is already respectful. No changes needed.
                </div>""", unsafe_allow_html=True)

        elif analyze_btn and not user_text.strip():
            st.markdown("""
            <div class="tc-card-accent-blue">
                <span style="font-size:0.88rem;color:#93c5fd;">Please enter some text before analyzing.</span>
            </div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="tc-placeholder">
                <div style="font-size:2rem;margin-bottom:0.75rem;">🔍</div>
                <div style="font-size:0.9rem;font-weight:500;color:#374151;margin-bottom:0.4rem;">No analysis yet</div>
                <div style="font-size:0.82rem;">
                    Enter text on the left and click<br>
                    <strong style="color:#6366f1;">Analyze & Neutralize</strong>
                </div>
            </div>""", unsafe_allow_html=True)


# ══ TAB 2 ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="tc-section-label">Batch Input</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.85rem;color:#6b7280;margin-bottom:0.75rem;">Enter one text per line. All will be analyzed simultaneously.</div>',
                unsafe_allow_html=True)

    batch_text = st.text_area(
        "Batch input", height=160,
        placeholder="You are such an idiot!\nHave a wonderful day!\nI hate everything about you...",
        label_visibility="collapsed", key="batch_input"
    )

    if st.button("Run Batch Analysis →", type="primary", key="batch_btn"):
        if batch_text.strip():
            texts = [t.strip() for t in batch_text.split('\n') if t.strip()]

            with st.spinner(f"Analyzing {len(texts)} texts..."):
                rows = []
                for text in texts:
                    r = pipeline.analyze(text)
                    rows.append({
                        'Text': text[:70] + ('...' if len(text) > 70 else ''),
                        'Status': '🔴 Toxic' if r['is_toxic'] else '🟢 Clean',
                        'Confidence': f"{r['confidence']:.0%}",
                        'Types': ', '.join(r['toxicity_types']) or '—',
                        'Neutralized': r['neutral_text'][:70] + ('...' if len(r['neutral_text']) > 70 else '')
                    })

            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=300)
            st.markdown("<hr style='margin:1.5rem 0;'>", unsafe_allow_html=True)

            toxic_n = sum(1 for r in rows if '🔴' in r['Status'])
            clean_n = len(rows) - toxic_n
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Analyzed", len(rows))
            m2.metric("Toxic", toxic_n)
            m3.metric("Clean", clean_n)
            m4.metric("Toxic Rate", f"{toxic_n/len(rows)*100:.1f}%")
        else:
            st.warning("Please enter at least one line of text.")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding-top:1.5rem;border-top:1px solid #1a1a2e;
     display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;">
    <div style="font-size:0.8rem;color:#374151;">
        <strong style="color:#6366f1;">ToxiClean</strong> · Built with Python, NLTK, scikit-learn & Streamlit
    </div>
    <div style="font-size:0.75rem;color:#374151;">NLP Project · For Educational Use</div>
</div>
""", unsafe_allow_html=True)
