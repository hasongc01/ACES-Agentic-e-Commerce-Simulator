import streamlit as st
import pandas as pd
import subprocess
from pathlib import Path
import re

# ======================================================
# STREAMLIT PAGE CONFIG
# ======================================================
st.set_page_config(page_title="Agent Shopping Simulator", layout="wide")

# ======================================================
# GLOBAL STYLE
# ======================================================
st.markdown(
    """
    <style>
        [data-testid="stAppViewBlockContainer"] {
            padding-top: 0.6rem;
            padding-bottom: 0.6rem;
        }

        div.stButton > button {
            font-size: 0.8rem;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
        }

        .product-title {
            font-size: 0.9rem;
            font-weight: 600;
            max-width: none;
            line-height: 1.2;
            display: -webkit-box;
            -webkit-line-clamp: 4;
            -webkit-box-orient: vertical;
            overflow: hidden;
            text-overflow: ellipsis;
            margin-bottom: 0.2rem;
        }

        .product-meta {
            font-size: 0.8rem;
            color: #4b5563;
            margin: 0.2rem 0 0.4rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# UI HEADER
# ======================================================
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:0.4rem; margin-bottom:0.2rem;">
        <span style="font-size:1.4rem;">🛒 <strong>Agent Shopping Simulator</strong></span>
    </div>
    <p style="font-size:0.85rem; color:#6b7280; margin-top:0;">
        1. Choose a product category & LLM model  
        2. You can use default prompt or write your own prompt
        3. Click <em>Run Simulator</em>  
        4. Agent's chosen product will be in red
        *Note: Results refresh automatically
    </p>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# CONSTANTS
# ======================================================
BASE_DIR = Path(__file__).resolve().parent
EXPERIMENT_LOGS_DIR = BASE_DIR / "experiment_logs"

DATASETS = {
    "fitness_watch": BASE_DIR / "datasets" / "fitness_watch.csv",
    "iphone_16_pro_cover": BASE_DIR / "datasets" / "iphone_16_pro_cover.csv",
    "mousepad": BASE_DIR / "datasets" / "mousepad.csv",
    "office_lamp": BASE_DIR / "datasets" / "office_lamp.csv",
    "stapler": BASE_DIR / "datasets" / "stapler.csv",
    "toilet_paper": BASE_DIR / "datasets" / "toilet_paper.csv",
    "toothpaste": BASE_DIR / "datasets" / "toothpaste.csv",
    "washing_machine": BASE_DIR / "datasets" / "washing_machine.csv",
}

MODEL_OPTIONS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-3.0-pro",
    "claude-3.5-sonnet",
    "claude-3.7-sonnet",
    "claude-4-sonnet",
    "claude-4.5-sonnet",
    "gpt-4.1",
    "gpt-4o",
    "gpt-5",
    "gpt-5.1"
]

DEFAULT_PROMPT = """You are my personal shopping assistant. Help me select a good product."""

# ======================================================
# UTIL: Render product image
# ======================================================
def render_product_image(url, highlight=False):
    border = (
        "3px solid #dc2626; background-color:#fef3c7;"
        if highlight
        else "1px solid #e5e7eb; background-color:#ffffff;"
    )

    if not url or not isinstance(url, str) or not url.startswith("http"):
        url = "https://via.placeholder.com/300"

    st.markdown(
        f"""
        <div style="
            background-color:#f3f4f6;
            border-radius:12px;
            padding:8px;
            display:flex;
            align-items:center;
            justify-content:center;
            height:180px;
            margin-bottom:0.5rem;
            border:{border};
        ">
            <img src="{url}" style="max-width:100%; max-height:100%; object-fit:contain;" />
        </div>
        """,
        unsafe_allow_html=True,
    )

# ======================================================
# UTIL: Run ACES
# ======================================================
def run_aces(local_dataset: str, model: str, prompt=None):
    cmd = [
        "uv", "run",
        str(BASE_DIR / "run.py"),
        "--runtime-type", "screenshot",
        "--local-dataset", str(local_dataset),
        "--include", model,
    ]

    if prompt and prompt.strip():
        cmd.extend(["--prompt-override", prompt.strip()])

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE_DIR))
    return result

# ======================================================
# UTIL: Find latest experiment CSV
# ======================================================
def get_latest_experiment_csv(dataset_name: str) -> Path | None:
    root = EXPERIMENT_LOGS_DIR / dataset_name
    if not root.exists():
        return None

    # Your real structure:
    # experiment_logs/<dataset>/<engine_run_id>/<dataset>/master_experiment_0/experiment_data.csv
    candidates = list(root.rglob("master_experiment_0/experiment_data.csv"))
    if not candidates:
        return None

    # newest first
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

# ======================================================
# SIDEBAR CONTROLS
# ======================================================
st.sidebar.title("Controls")

categories = list(DATASETS.keys())
pretty_labels = [c.replace("_", " ") for c in categories]
cat_map = dict(zip(pretty_labels, categories))

label_selected = st.sidebar.selectbox("Product category", pretty_labels)
dataset_selected = DATASETS[cat_map[label_selected]]

model_selected = st.sidebar.selectbox("VLM model", MODEL_OPTIONS)

user_prompt = st.sidebar.text_area(
    "Custom shopping prompt (optional)",
    value=DEFAULT_PROMPT,
    height=300,
)

run_button = st.sidebar.button("🚀 Run Simulator", type="primary")

# ======================================================
# RESET STATE WHENEVER ANY INPUT CHANGES
# ======================================================
if "prev_state" not in st.session_state:
    st.session_state["prev_state"] = (None, None, None)

state_now = (dataset_selected, model_selected, user_prompt)
if state_now != st.session_state["prev_state"]:
    st.session_state["df"] = None
    st.session_state["prev_state"] = state_now

# ======================================================
# RUN EXPERIMENT
# ======================================================
if run_button:
    with st.status("Your Agent is Shopping! Please wait patiently ...", expanded=True):
        res = run_aces(dataset_selected, model_selected, prompt=user_prompt)
        # st.markdown("### Output:")
        # st.code(res.stdout + "\n" + res.stderr)

    csv_path = get_latest_experiment_csv(cat_map[label_selected])
    if not csv_path:
        st.error("No experiment_data.csv found — ACES may not have produced output.")
        st.stop()

    df = pd.read_csv(csv_path)
    st.session_state["df"] = df
    st.rerun()

# ======================================================
# IF NO DATA: STOP
# ======================================================
if "df" not in st.session_state or st.session_state["df"] is None:
    st.info("Run the simulator from the sidebar.")
    st.stop()

df = st.session_state["df"]

# sort by assigned position
if "assigned_position" in df.columns:
    df = df.sort_values("assigned_position").reset_index(drop=True)

# ======================================================
# DETECT AGENT SELECTION
# ======================================================
agent_sku = None
if "selected" in df.columns:
    picked = df[df["selected"] == 1]
    if not picked.empty:
        agent_sku = picked.iloc[0]["sku"]

# fallback
if agent_sku is None:
    agent_sku = df.iloc[0]["sku"]


# ======================================================
# GRID VIEW WITH AGENT HIGHLIGHT
# ======================================================
sku_order_current = df["sku"].tolist()
sku_to_row = {row["sku"]: row for _, row in df.iterrows()}
agent_selected_sku = agent_sku   # unify variable names


num_cols = 4
cols = st.columns(num_cols)

def rating_to_stars(rating: float) -> str:
    """Return a 5-star string like ★★★★☆ based on the numeric rating."""
    if rating is None:
        return "☆☆☆☆☆"
    try:
        r = float(rating)
    except (TypeError, ValueError):
        return "☆☆☆☆☆"

    # Buckets: 2.5–3.4 → 3 stars, 3.5–4.4 → 4, 4.5–5 → 5
    if r >= 4.5:
        filled = 5
    elif r >= 3.5:
        filled = 4
    elif r >= 2.5:
        filled = 3
    elif r > 0:
        filled = 2
    else:
        filled = 0

    total = 5
    return "★" * filled + "☆" * (total - filled)



for i, sku in enumerate(sku_order_current):
    if sku not in sku_to_row:
        continue

    row = sku_to_row[sku]
    is_agent_pick = (agent_selected_sku is not None and sku == agent_selected_sku)

    with cols[i % num_cols]:
        # Outer card: this is what gets the big red border for the selected product
        outer_style = (
            "border: 3px solid #dc2626; background-color:#fef3c7; "
            "border-radius: 16px; padding: 10px; margin-bottom: 12px;"
            if is_agent_pick
            else
            "border: 1px solid #e5e7eb; background-color:#ffffff; "
            "border-radius: 16px; padding: 10px; margin-bottom: 12px;"
        )
        # st.markdown(f'<div style="{outer_style}">', unsafe_allow_html=True)

        # --- IMAGE AREA (just the image, no extra empty grey box) ---
        render_product_image(row.get("image_url"), highlight=is_agent_pick)

        # --- WHITE CONTENT AREA UNDER IMAGE ---
        st.markdown(
            '<div style="background-color:#ffffff; padding:8px 6px 10px 6px; border-radius:12px;">',
            unsafe_allow_html=True,
        )

        # Title (up to 4 lines, ellipsis handled by CSS class)
        full_title = str(row["title"])
        title_color = "#dc2626" if is_agent_pick else "#111827"
        st.markdown(
            f"""
            <div class="product-title" style="color:{title_color};">
                {full_title}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Stars + (# reviews)
        rating = row.get("rating", None)
        rating_count = row.get("rating_count", None)
        if rating is not None and pd.notna(rating):
            stars = rating_to_stars(rating)
            reviews_part = ""
            if rating_count is not None and pd.notna(rating_count):
                reviews_part = f" ({int(rating_count)})"
            st.markdown(
                f'<p class="product-meta">{stars} {float(rating):.1f}{reviews_part}</p>',
                unsafe_allow_html=True,
            )

        # Price row
        price = row.get("price", None)
        if price is not None and pd.notna(price):
            st.markdown(
                f'<p class="product-meta"><strong>${float(price):.2f}</strong></p>',
                unsafe_allow_html=True,
            )

        # --- Bottom row: Add to Cart (left) + pill (right) ---
        low_stock = bool(row.get("low_stock", False))
        stock_quantity = row.get("stock_quantity", None)
        overall_pick = bool(row.get("overall_pick", False))
        sponsored = bool(row.get("sponsored", False))

        # Decide which pill to show (priority: low_stock > overall_pick > sponsored)
        pill_html = ""
        if low_stock and stock_quantity is not None and pd.notna(stock_quantity):
            pill_html = (
                '<span style="background-color:#f97316; color:white; '
                'font-size:0.7rem; padding:2px 8px; border-radius:999px; '
                'white-space:nowrap;">Only '
                f'{int(stock_quantity)} left</span>'
            )
        elif overall_pick:
            pill_html = (
                '<span style="background-color:#1d4ed8; color:white; '
                'font-size:0.7rem; padding:2px 8px; border-radius:999px; '
                'white-space:nowrap;">Overall pick</span>'
            )
        elif sponsored:
            pill_html = (
                '<span style="background-color:#6b7280; color:white; '
                'font-size:0.7rem; padding:2px 8px; border-radius:999px; '
                'white-space:nowrap;">Sponsored</span>'
            )

        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:0.5rem;">
                <button type="button" style="
                    text-decoration:none;
                    background-color:#f59e0b;
                    color:#111827;
                    padding:6px 14px;
                    border-radius:999px;
                    font-size:0.8rem;
                    font-weight:600;
                ">
                    Add to Cart
                </button>
                {pill_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Close white body + outer card
        st.markdown("</div>", unsafe_allow_html=True)  # white body
        # st.markdown("</div>", unsafe_allow_html=True)  # outer card
