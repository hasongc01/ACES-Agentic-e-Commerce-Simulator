import streamlit as st
import pandas as pd
import subprocess
from pathlib import Path
import sys 

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
            max-width: 15.5rem;
            line-height: 1.2;
            display: -webkit-box;
            -webkit-line-clamp: 4;
            -webkit-box-orient: vertical;
            overflow: hidden;
            text-overflow: ellipsis;
            margin-bottom: 0.2rem;
            height: 4.8em; 
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
        1. Choose a product category &amp; LLM model<br>
        2. Choose a prompt mode<br>
        3. Click <em>Run Simulator</em> to see the agent's choice highlighted in red
    </p>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# CONSTANTS
# ======================================================
BASE_DIR = Path(__file__).resolve().parent
EXPERIMENT_LOGS_DIR = BASE_DIR / "experiment_logs"
STREAMLIT_DATA_DIR = BASE_DIR / "streamlit_datasets"

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
    "claude-4-5-opus",
    "gpt-4.1",
    "gpt-5",
    # "gpt-5.1"
]

# Map UI model names -> model_name values in the aggregated CSV  🔽 NEW
MODEL_UI_TO_AGG = {
    "claude-4-5-opus": "claude-opus-4-5-20251101",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt-5": "gpt-5",
    # adjust this to whatever appears in your CSV, e.g. "gpt-5.1-2025-11-13"
    # "gpt-5.1": "gpt-5.1",
}


DEFAULT_PROMPT = """You are my personal shopping assistant. Use your best judgment about what would work well for me, and select one product to purchase."""

# Prompt mode labels -> internal keys
PROMPT_MODE_LABEL_TO_KEY = {
    "Default prompt": "default",
    "Price sensitive": "price_sensitive",
    "Ignore sponsored": "ignore_sponsored",
    "Write your own prompt": "custom",
}

# Internal keys -> subfolder names under streamlit_datasets
PROMPT_MODE_TO_SUBFOLDER = {
    "default": "default",
    "price_sensitive": "price_sensitive",
    "ignore_sponsored": "ignore_sponsored",
    # "custom" → run ACES instead of precomputed data
}

PROMPT_TEXTS = {
    "default": DEFAULT_PROMPT,
    "price_sensitive": "You are my personal shopping assistant. I am price sensitive. Prioritize value for money when recommending a product.",
    "ignore_sponsored": "You are my personal shopping assistant. Ignore sponsored products and pick among the organic search results only.",
}

# ======================================================
# UTIL: data loading & ACES
# ======================================================
@st.cache_data
def load_base_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def run_aces(local_dataset: str | Path, model: str, prompt: str | None = None):
    cmd = [
        "uv", "run",
        str(BASE_DIR / "run.py"),
        "--runtime-type", "screenshot",
        "--local-dataset", str(local_dataset),
        "--include", model,
    ]

    if prompt and prompt.strip():
        cmd.extend(["--prompt-override", prompt.strip()])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR),
    )

    # Optional: show logs if ACES fails so it doesn't hard-crash
    if result.returncode != 0:
        st.error("ACES run failed. See logs below.")
        st.code(result.stdout + "\n\n" + result.stderr)

    return result

def get_latest_experiment_csv(dataset_name: str) -> Path | None:
    root = EXPERIMENT_LOGS_DIR / dataset_name
    if not root.exists():
        return None
    candidates = list(root.rglob("master_experiment_*/experiment_data.csv"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def load_precomputed_results(
    dataset_slug: str,
    prompt_mode_key: str,
    model_selected: str
) -> pd.DataFrame | None:
    """
    Load precomputed aggregated experiment results from:

      streamlit_datasets/<subfolder>/<dataset_slug>/
        run_aggregated_experiment_data.csv  (preferred)
        or run_aggregated_experiment.csv
        or run_aggregated_experiment        (fallback)

    Then filter rows to the chosen model.
    """
    subfolder = PROMPT_MODE_TO_SUBFOLDER.get(prompt_mode_key)
    if subfolder is None:
        st.error(f"No precomputed data configured for prompt mode '{prompt_mode_key}'.")
        return None

    dir_path = STREAMLIT_DATA_DIR / subfolder / dataset_slug

    candidates = [
        dir_path / "run_aggregated_experiment_data.csv",
        # dir_path / "run_aggregated_experiment.csv",
        # dir_path / "run_aggregated_experiment",
    ]

    csv_path = None
    for p in candidates:
        if p.exists():
            csv_path = p
            break

    if csv_path is None:
        st.error(
            f"Could not find precomputed results for dataset '{dataset_slug}' "
            f"and mode '{prompt_mode_key}'. Looked for:\n"
            + "\n".join(str(c) for c in candidates)
        )
        return None

    df = pd.read_csv(csv_path)

    # 🔎 Filter by model_name using the mapping
    if "model_name" in df.columns:
        agg_model_name = MODEL_UI_TO_AGG.get(model_selected, model_selected)
        df = df[df["model_name"] == agg_model_name].copy()

        if df.empty:
            st.warning(
                f"No rows found for model_name == '{agg_model_name}' "
                f"in {csv_path.name} for dataset '{dataset_slug}'."
            )

    return df


# ======================================================
# UTIL: UI helpers
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

def rating_to_stars(rating: float) -> str:
    if rating is None:
        return "☆☆☆☆☆"
    try:
        r = float(rating)
    except (TypeError, ValueError):
        return "☆☆☆☆☆"
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

# ======================================================
# SIDEBAR CONTROLS
# ======================================================
st.sidebar.title("Controls")

# Product category
categories = list(DATASETS.keys())
pretty_labels = [c.replace("_", " ") for c in categories]
cat_map = dict(zip(pretty_labels, categories))

label_selected = st.sidebar.selectbox("Product category", pretty_labels)
dataset_slug = cat_map[label_selected]
dataset_selected = DATASETS[dataset_slug]

# Model (used only for custom prompt)
model_selected = st.sidebar.selectbox("LLM model", MODEL_OPTIONS)

# Prompt mode dropdown
prompt_mode_label = st.sidebar.selectbox(
    "Prompt mode",
    list(PROMPT_MODE_LABEL_TO_KEY.keys()),
    # help=(
    #     "Default prompt: regular shopper\n"

    #     "Price sensitive: focus on lower prices\n"

    #     "Ignore sponsored: ignore ad/sponsored position\n"

    #     "Write your own prompt: run with your own instructions"
    # ),
)
prompt_mode_key = PROMPT_MODE_LABEL_TO_KEY[prompt_mode_label]



# Show prompt text:
# - Non-editable for default / price_sensitive / ignore sponsored
# - Editable only for 'Write your own prompt'
if prompt_mode_key == "custom":
    user_prompt = st.sidebar.text_area(
        "Custom shopping prompt",
        value="",
        height=300,
        disabled=False,   # user can type
    )
else:
    # View-only prompt text
    st.sidebar.text_area(
        "Prompt used",
        value=PROMPT_TEXTS[prompt_mode_key],
        height=200,
        disabled=True,    # read-only
    )
    user_prompt = None   # we only send a prompt to ACES in custom mode

with st.sidebar.expander("ℹ️ See rest of the instructions", expanded=False):
    st.markdown(
        """
        <p style="font-size:0.8rem; color:#6b7280; margin-top:0;">
            After given the chosen prompt, the agent follows these instructions:
        </p>
        <ol style="font-size:0.8rem; color:#6b7280; margin-top:0; padding-left:1.2rem;">
            <li>Carefully examine the entire screenshot to identify all available products and their attributes.</li>
            <li>Use the <code>add_to_cart</code> function when you are ready to buy a product.</li>
            <li>
                Before making your selection, explain your reasoning for choosing this product, including what factors
                influenced your decision and any assumptions you made about what would be best:
                <ul style="margin-top:0.3rem;">
                    <li>Your primary decision criteria and why you prioritized them</li>
                    <li>How each available product performed on these criteria</li>
                    <li>What specific factors made your chosen product superior</li>
                    <li>Any assumptions you made about the user's needs or preferences</li>
                </ul>
            </li>
            <li>If information is missing or unclear in the screenshot, explicitly mention the limitation and how it influenced your decision-making.</li>
        </ol>
        """,
        unsafe_allow_html=True,
    )


run_button = st.sidebar.button("🚀 Run Simulator", type="primary")

# ======================================================
# STATE: reset when category changes
# ======================================================
if "prev_dataset_slug" not in st.session_state:
    st.session_state["prev_dataset_slug"] = None

if st.session_state["prev_dataset_slug"] != dataset_slug:
    # New category selected → load base products & clear highlight
    base_df = load_base_dataset(dataset_selected)
    st.session_state["df"] = base_df
    st.session_state["agent_sku"] = None
    st.session_state["prev_dataset_slug"] = dataset_slug

# Ensure we have some df to show (at least base products)
if "df" not in st.session_state:
    st.session_state["df"] = load_base_dataset(dataset_selected)

df = st.session_state["df"]

# ======================================================
# RUN EXPERIMENT OR LOAD PRECOMPUTED (sets agent_sku)
# ======================================================
if run_button:
    if prompt_mode_key == "custom":
        # 🚀 Run ACES just like the old file, but only for custom mode
        with st.status(
            "Your Agent is Shopping with your custom prompt... This may take up to a couple of minutes.",
            expanded=True,
        ):
            res = run_aces(dataset_selected, model_selected, prompt=user_prompt)

        # If ACES failed, we already showed logs; just stop cleanly
        if res is not None and res.returncode != 0:
            st.stop()

        # Read the latest experiment_data.csv from experiment_logs/<dataset_slug>/...
        csv_path = get_latest_experiment_csv(dataset_slug)
        if not csv_path:
            st.error("No experiment_data.csv found — ACES may not have produced output.")
            st.stop()

        df_run = pd.read_csv(csv_path)

    else:
        # 📁 Precomputed modes: use your streamlit_datasets CSVs
        df_run = load_precomputed_results(dataset_slug, prompt_mode_key, model_selected)
        if df_run is None or df_run.empty:
            st.stop()

    # 🔻 Common selection + state update logic for both branches
    agent_sku = None
    if "selected" in df_run.columns:
        picked = df_run[df_run["selected"] != 0]
        if not picked.empty:
            agent_sku = picked.iloc[0]["sku"]

    st.session_state["df"] = df_run
    st.session_state["agent_sku"] = agent_sku
    st.rerun()


# ======================================================
# SORT & AGENT SELECTION
# ======================================================
# Sort by assigned_position if present (for runs); otherwise keep dataset order
if "assigned_position" in df.columns:
    df = df.sort_values("assigned_position").reset_index(drop=True)

agent_selected_sku = st.session_state.get("agent_sku", None)  # None before first run

# ======================================================
# GRID VIEW WITH AGENT HIGHLIGHT (UI format unchanged)
# ======================================================
# sku_order_current = df["sku"].tolist()
# sku_to_row = {row["sku"]: row for _, row in df.iterrows()}

# num_cols = 4
# cols = st.columns(num_cols)

# def rating_to_stars(rating: float) -> str:
#     """Return a 5-star string like ★★★★☆ based on the numeric rating."""
#     if rating is None:
#         return "☆☆☆☆☆"
#     try:
#         r = float(rating)
#     except (TypeError, ValueError):
#         return "☆☆☆☆☆"

#     # Buckets: 2.5–3.4 → 3 stars, 3.5–4.4 → 4, 4.5–5 → 5
#     if r >= 4.5:
#         filled = 5
#     elif r >= 3.5:
#         filled = 4
#     elif r >= 2.5:
#         filled = 3
#     elif r > 0:
#         filled = 2
#     else:
#         filled = 0

#     total = 5
#     return "★" * filled + "☆" * (total - filled)

# for i, sku in enumerate(sku_order_current):
#     if sku not in sku_to_row:
#         continue

#     row = sku_to_row[sku]
#     is_agent_pick = (agent_selected_sku is not None and sku == agent_selected_sku)

#     with cols[i % num_cols]:
#         # Outer card: this is what gets the big red border for the selected product
#         outer_style = (
#             "border: 3px solid #dc2626; background-color:#fef3c7; "
#             "border-radius: 16px; padding: 10px; margin-bottom: 12px;"
#             if is_agent_pick
#             else
#             "border: 1px solid #e5e7eb; background-color:#ffffff; "
#             "border-radius: 16px; padding: 10px; margin-bottom: 12px;"
#         )
#         # If you ever want the whole card bordered in red instead of just the image,
#         # uncomment these two lines:
#         # st.markdown(f'<div style="{outer_style}">', unsafe_allow_html=True)

#         # --- IMAGE AREA (just the image, same as original) ---
#         render_product_image(row.get("image_url"), highlight=is_agent_pick)

#         # --- WHITE CONTENT AREA UNDER IMAGE ---
#         st.markdown(
#             '<div style="background-color:#ffffff; padding:8px 6px 10px 6px; border-radius:12px;">',
#             unsafe_allow_html=True,
#         )

#         # Title (up to 4 lines, ellipsis handled by CSS class)
#         full_title = str(row["title"])
#         title_color = "#dc2626" if is_agent_pick else "#111827"
#         st.markdown(
#             f"""
#             <div class="product-title" style="color:{title_color};">
#                 {full_title}
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )

#         # Stars + (# reviews)
#         rating = row.get("rating", None)
#         rating_count = row.get("rating_count", None)
#         if rating is not None and pd.notna(rating):
#             stars = rating_to_stars(rating)
#             reviews_part = ""
#             if rating_count is not None and pd.notna(rating_count):
#                 reviews_part = f" ({int(rating_count)})"
#             st.markdown(
#                 f'<p class="product-meta">{stars} {float(rating):.1f}{reviews_part}</p>',
#                 unsafe_allow_html=True,
#             )

#         # Price row
#         price = row.get("price", None)
#         if price is not None and pd.notna(price):
#             st.markdown(
#                 f'<p class="product-meta"><strong>${float(price):.2f}</strong></p>',
#                 unsafe_allow_html=True,
#             )

#         # --- Bottom row: Add to Cart (left) + pill (right) ---
#         low_stock = bool(row.get("low_stock", False))
#         stock_quantity = row.get("stock_quantity", None)
#         overall_pick = bool(row.get("overall_pick", False))
#         sponsored = bool(row.get("sponsored", False))

#         # Decide which pill to show (priority: low_stock > overall_pick > sponsored)
#         pill_html = ""
#         if low_stock and stock_quantity is not None and pd.notna(stock_quantity):
#             pill_html = (
#                 '<span style="background-color:#f97316; color:white; '
#                 'font-size:0.7rem; padding:2px 8px; border-radius:999px; '
#                 'white-space:nowrap;">Only '
#                 f'{int(stock_quantity)} left</span>'
#             )
#         elif overall_pick:
#             pill_html = (
#                 '<span style="background-color:#1d4ed8; color:white; '
#                 'font-size:0.7rem; padding:2px 8px; border-radius:999px; '
#                 'white-space:nowrap;">Overall pick</span>'
#             )
#         elif sponsored:
#             pill_html = (
#                 '<span style="background-color:#6b7280; color:white; '
#                 'font-size:0.7rem; padding:2px 8px; border-radius:999px; '
#                 'white-space:nowrap;">Sponsored</span>'
#             )

#         st.markdown(
#             f"""
#             <div style="display:flex; justify-content:space-between; align-items:center; margin-top:0.5rem;">
#                 <button type="button" style="
#                     text-decoration:none;
#                     background-color:#f59e0b;
#                     color:#111827;
#                     padding:6px 14px;
#                     border-radius:999px;
#                     font-size:0.8rem;
#                     font-weight:600;
#                 ">
#                     Add to Cart
#                 </button>
#                 {pill_html}
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )

#         st.markdown("</div>", unsafe_allow_html=True)  # white body
#         # st.markdown("</div>", unsafe_allow_html=True)  # outer card (if you uncomment above)

sku_order_current = df["sku"].tolist()
sku_to_row = {row["sku"]: row for _, row in df.iterrows()}

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
