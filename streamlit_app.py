import streamlit as st
import torch
import time

from simplify import TextSimplifier
from translate import IndicTranslator

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Text Simplification + Translator",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

CARD_H = 530          # ← both panels share this height
# Fixed overhead inside right card:
#   card padding 32 + subheader 42 + simp-label 24 + divider 30 + select-row 48 + trans-label 24 = 200
OUT_BOX_H = (CARD_H - 240) // 2   # each output box gets equal share

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*, *::before, *::after {{ font-family: 'Inter', sans-serif; box-sizing: border-box; }}

/* ── App chrome ── */
[data-testid="stAppViewContainer"] {{ background-color: #0b1120; color: #e2e8f0; }}
[data-testid="stHeader"]           {{ background-color: transparent; }}
.block-container {{
    padding-top: 1.4rem !important;
    padding-bottom: 0.4rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 100% !important;
}}

/* ── Align column tops ── */
[data-testid="stHorizontalBlock"] {{
    align-items: flex-start !important;
    gap: 1.25rem !important;
}}

/* ── Card styling ── */
[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 12px !important;
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
}}

/* ── Title ── */
h1 {{
    text-align: center;
    font-weight: 700;
    color: #fff;
    font-size: 2rem !important;
    margin: 0 0 0.15rem !important;
    padding: 0 !important;
}}
.subtitle {{
    text-align: center;
    color: #94a3b8;
    font-size: 0.9rem;
    margin: 0 0 1rem !important;
}}

/* ── Textarea ── */
.stTextArea textarea {{
    background-color: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
    font-size: 1rem !important;
    resize: none !important;
}}
.stTextArea textarea:focus {{
    border-color: #334155 !important;
    box-shadow: none !important;
    outline: none !important;
}}

/* ── Output elements ── */
.out-label {{
    font-weight: 600;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 6px;
}}
.out-box {{
    background-color: #0f172a;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 14px 16px;
    color: #cbd5e1;
    line-height: 1.65;
    font-size: 1rem;
    min-height: 130px;
}}
.out-divider {{
    border: none;
    border-top: 1px solid #263347;
    margin: 14px 0 12px 0;
}}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {{
    background-color: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
    font-size: 0.9rem !important;
}}

/* ── Subheader ── */
h3 {{
    margin-bottom: 10px !important;
    font-size: 1.05rem !important;
    color: #fff !important;
    font-weight: 600 !important;
}}

/* ── Primary button ── */
.stButton button {{
    background-color: #3b82f6 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    height: 46px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}}
.stButton button:hover {{
    background-color: #2563eb !important;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
for key in (
    "input_text",
    "simplified_text",
    "translated_text",
    "simp_times",
    "trans_times",
):
    if key not in st.session_state:
        st.session_state[key] = [] if key.endswith("_times") else ""


def _avg(values):
    return sum(values) / len(values) if values else None


def _sync_cuda():
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def _fmt_sec(seconds):
    return f"{seconds:.1f}s" if seconds < 60 else f"{seconds/60:.1f}m"


def _eta_bounds(estimate, samples):
    if samples <= 0:
        lo_mul, hi_mul = 0.75, 1.80
    elif samples < 3:
        lo_mul, hi_mul = 0.80, 1.50
    elif samples < 8:
        lo_mul, hi_mul = 0.85, 1.35
    else:
        lo_mul, hi_mul = 0.90, 1.20
    return estimate * lo_mul, estimate * hi_mul


def _fmt_range(lo, hi):
    return f"{_fmt_sec(lo)}-{_fmt_sec(hi)}"


def _estimate_eta(input_text):
    # Blend learned timings with a conservative length-based heuristic.
    words = len((input_text or "").strip().split())
    simp_avg = _avg(st.session_state.simp_times)
    trans_avg = _avg(st.session_state.trans_times)

    simp_fallback = min(240.0, 25.0 + (0.45 * words))
    trans_fallback = min(180.0, 18.0 + (0.30 * words))

    simp_eta = max(simp_avg, 0.60 * simp_fallback) if simp_avg is not None else simp_fallback
    trans_eta = max(trans_avg, 0.60 * trans_fallback) if trans_avg is not None else trans_fallback

    simp_lo, simp_hi = _eta_bounds(simp_eta, len(st.session_state.simp_times))
    trans_lo, trans_hi = _eta_bounds(trans_eta, len(st.session_state.trans_times))
    total_lo, total_hi = simp_lo + trans_lo, simp_hi + trans_hi

    return (simp_eta, simp_lo, simp_hi), (trans_eta, trans_lo, trans_hi), (total_lo, total_hi)

# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models — first run only…")
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TextSimplifier(device=device), IndicTranslator(device=device)

try:
    simplifier, translator = load_models()
    languages = translator.get_supported_languages()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# ─────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────
st.markdown("<h1>Text Simplification + Indic Translation</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enter complex English text, simplify it, and translate into an Indic language.</p>", unsafe_allow_html=True)

def _sync():
    st.session_state.input_text = st.session_state._ti

# ─────────────────────────────────────────────
# Two equal-height panels
# ─────────────────────────────────────────────
col_left, col_right = st.columns(2, gap="large")

# ── LEFT ──
with col_left:
    with st.container(border=True, height=CARD_H):
        st.subheader("Input Text")
        # textarea height = CARD_H minus subheader (~46px) minus card padding (~32px)
        st.text_area(
            "input",
            height=CARD_H - 46 - 32,
            label_visibility="collapsed",
            placeholder="Enter complex English text here…",
            value=st.session_state.input_text,
            key="_ti",
            on_change=_sync,
        )

# ── RIGHT ──
with col_right:
    with st.container(border=True, height=CARD_H):
        st.subheader("Output")

        # Simplified English
        st.markdown("<div class='out-label'>Simplified English</div>", unsafe_allow_html=True)
        simp_slot = st.empty()
        simp_slot.markdown(
            f"<div class='out-box' style='height:{OUT_BOX_H}px;overflow-y:auto;'>{st.session_state.simplified_text or 'Simplified text will appear here…'}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<hr class='out-divider'>", unsafe_allow_html=True)

        # Translated Output
        lc, rc = st.columns([1, 1], vertical_alignment="center")
        with lc:
            st.markdown("<div class='out-label'>Translated Output</div>", unsafe_allow_html=True)
        with rc:
            target_language = st.selectbox(
                "Language",
                options=languages,
                index=languages.index("Hindi") if "Hindi" in languages else 0,
                label_visibility="collapsed",
            )
        trans_slot = st.empty()
        trans_slot.markdown(
            f"<div class='out-box' style='height:{OUT_BOX_H}px;overflow-y:auto;'>{st.session_state.translated_text or 'Translated text will appear here…'}</div>",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────
# Run button
# ─────────────────────────────────────────────
st.write("")
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    run = st.button("Simplify & Translate", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────
if run:
    if not st.session_state.input_text.strip():
        st.warning("Please enter some text first.")
    else:
        simp_eta, trans_eta, total_eta = _estimate_eta(st.session_state.input_text)
        simp_mid, simp_lo, simp_hi = simp_eta
        trans_mid, trans_lo, trans_hi = trans_eta
        total_lo, total_hi = total_eta
        simp_slot.markdown(
            f"<div class='out-box' style='height:{OUT_BOX_H}px;overflow-y:auto;color:#64748b;font-style:italic;'>Simplifying English… ETA ~{_fmt_range(simp_lo, simp_hi)}</div>",
            unsafe_allow_html=True,
        )

        _sync_cuda()
        simp_start = time.perf_counter()
        st.session_state.simplified_text = simplifier.simplify_text(
            st.session_state.input_text.strip()
        )
        _sync_cuda()
        simp_elapsed = time.perf_counter() - simp_start

        trans_slot.markdown(
            f"<div class='out-box' style='height:{OUT_BOX_H}px;overflow-y:auto;color:#64748b;font-style:italic;'>Translating to {target_language}… ETA ~{_fmt_range(trans_lo, trans_hi)}</div>",
            unsafe_allow_html=True,
        )

        _sync_cuda()
        trans_start = time.perf_counter()
        st.session_state.translated_text = translator.translate(
            st.session_state.simplified_text, target_language
        )
        _sync_cuda()
        trans_elapsed = time.perf_counter() - trans_start

        # Keep a short history so ETA adapts without drifting too much.
        st.session_state.simp_times = (st.session_state.simp_times + [simp_elapsed])[-20:]
        st.session_state.trans_times = (st.session_state.trans_times + [trans_elapsed])[-20:]

        st.session_state.last_used_language = target_language
        st.rerun()
