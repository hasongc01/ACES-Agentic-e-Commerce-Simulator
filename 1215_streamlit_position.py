import streamlit as st
import pandas as pd
import subprocess
from pathlib import Path
import sys
import shutil

# ======================================================
# STREAMLIT PAGE CONFIG
# ======================================================
st.set_page_config(page_title="Agentic Shopping Simulator", layout="wide")

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

        .instructions-text {
            font-family: var(--font);
            font-size: 1.2rem;
            color: #000000;
            margin-top: 0;
        }

        .mock-marketplace-title {
            font-family: var(--font);
            font-size: 1.2rem;
            font-weight: 600;
            text-align: center;
            margin: 0 0 0.75rem 0;
        }

        /* Grey background for the bordered container that wraps all 8 products */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #f3f4f6 !important;
            border-radius: 12px;
            padding: 0.75rem 0.75rem 1rem 0.75rem;
            border: 1px solid #e5e7eb;
        }

        /* Make the inner content transparent so the grey shows through */
        [data-testid="stVerticalBlockBorderWrapper"] > div {
            background-color: transparent !important;
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
        <span style="
            font-size:1.4rem;
            font-family: var(--font);
            color: #000000;
        ">
            🛒 <strong>Agentic Shopping Simulator</strong>
        </span>
    </div>

    <p class="instructions-text">
        1. Choose a product category &amp; Gen AI model<br>
        2. Choose a prompt mode<br>
        3. Click <em>Run Simulator</em> to see the agent's choice highlighted in red<br>
        4. Optional: Click <em> Shuffle listing positions </em> to explore how listing positions affect the agent's choice
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
    "gpt-4.1",
    "gpt-5",
    "claude-4-5-opus",
    # "gpt-5.1"
]

# Map UI model names -> model_name values in the aggregated CSV
MODEL_UI_TO_AGG = {
    "claude-4-5-opus": "claude-opus-4-5-20251101",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt-5": "gpt-5",
    # "gpt-5.1": "gpt-5.1-2025-11-13",
}

DEFAULT_PROMPT = """You are my personal shopping assistant. Use your best judgment about what would work well for me, and select one product to purchase."""

PROMPT_MODE_LABEL_TO_KEY = {
    "Default prompt": "default",
    "Price sensitive": "price_sensitive",
    "Ignore sponsored": "ignore_sponsored",
    "Write your own prompt": "custom",
}

PROMPT_MODE_TO_SUBFOLDER = {
    "default": "default",
    "price_sensitive": "price_sensitive",
    "ignore_sponsored": "ignore_sponsored",
    # custom → run ACES instead of precomputed data
}

PROMPT_TEXTS = {
    "default": DEFAULT_PROMPT,
    "price_sensitive": "You are my personal shopping assistant. I am price sensitive. Prioritize value for money when recommending a product.",
    "ignore_sponsored": "You are my personal shopping assistant. Ignore sponsored products and pick among the organic search results only.",
}

# ======================================================
# STATE: random-position dataset switching (Default prompt only)
#   rand_variant: 0=original, 1=random dataset 1, 2=random dataset 2
#   rand_clicks: number of successful clicks (0..2)
# ======================================================
if "rand_variant" not in st.session_state:
    st.session_state["rand_variant"] = 0
# if "rand_clicks" not in st.session_state:
#     st.session_state["rand_clicks"] = 0

# ======================================================
# UTIL: data loading & ACES
# ======================================================
@st.cache_data
def load_base_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def run_aces(local_dataset: str | Path, model: str, prompt: str | None = None) -> int:
    """
    Run ACES via uv + screenshot runtime, quietly:
    - no backend logs are shown in the Streamlit UI
    - just return the process return code
    """
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
        cwd=str(BASE_DIR),
        text=True,
        capture_output=True,
    )
    return result.returncode


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
    model_selected: str,
    rand_variant: int = 0,  # 0=original, 1/2=random_position datasets
) -> pd.DataFrame | None:
    """
    Original (rand_variant=0):
      streamlit_datasets/<subfolder>/<dataset_slug>/run_aggregated_experiment_data.csv

    Random position (rand_variant=1 or 2), per-category:
      streamlit_datasets/<subfolder>/<dataset_slug>/random_position/run_aggregated_experiment_data1.csv
      streamlit_datasets/<subfolder>/<dataset_slug>/random_position/run_aggregated_experiment_data2.csv
    """
    subfolder = PROMPT_MODE_TO_SUBFOLDER.get(prompt_mode_key)
    if subfolder is None:
        st.error(f"No precomputed data configured for prompt mode '{prompt_mode_key}'.")
        return None

    base_dir = STREAMLIT_DATA_DIR / subfolder / dataset_slug

    if rand_variant in (1, 2):
        dir_path = base_dir / "random_position"
        candidates = [
            dir_path / f"run_aggregated_experiment_data{rand_variant}.csv",
            dir_path / f"run_aggregated_experiment_data{rand_variant}",  # optional no-ext fallback
        ]
    else:
        dir_path = base_dir
        candidates = [
            dir_path / "run_aggregated_experiment_data.csv",
            dir_path / "run_aggregated_experiment_data",  # optional no-ext fallback
        ]

    csv_path = None
    for p in candidates:
        if p.exists():
            csv_path = p
            break

    if csv_path is None:
        st.error(
            f"Could not find precomputed results for dataset '{dataset_slug}' "
            f"and mode '{prompt_mode_key}' (rand_variant={rand_variant}). Looked for:\n"
            + "\n".join(str(c) for c in candidates)
        )
        return None

    df = pd.read_csv(csv_path)

    # Filter by model_name using mapping
    if "model_name" in df.columns:
        agg_model_name = MODEL_UI_TO_AGG.get(model_selected, model_selected)
        df = df[df["model_name"] == agg_model_name].copy()

        if df.empty:
            st.warning(
                f"No rows found for model_name == '{agg_model_name}' "
                f"in {csv_path.name} for dataset '{dataset_slug}'."
            )

    return df


def extract_agent_pick(df_run: pd.DataFrame | None) -> tuple[str | None, str | None]:
    """Return (agent_sku, agent_title) from df_run."""
    if df_run is None or df_run.empty:
        return None, None

    if "selected" in df_run.columns:
        picked = df_run[df_run["selected"] != 0]
        if not picked.empty:
            return picked.iloc[0].get("sku", None), picked.iloc[0].get("title", None)

    return None, None


# ======================================================
# UTIL: UI helpers
# ======================================================
def render_product_image(url, highlight=False):
    border = (
        "3px solid #dc2626; background-color:#fef3c7;"
        if highlight
        else "1px solid #e5e7eb"
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
    return "★" * filled + "☆" * (5 - filled)


# ======================================================
# SIDEBAR CONTROLS
# ======================================================
st.sidebar.title("Controls")

categories = list(DATASETS.keys())
pretty_labels = [c.replace("_", " ") for c in categories]
cat_map = dict(zip(pretty_labels, categories))

label_selected = st.sidebar.selectbox("Product category", pretty_labels)
dataset_slug = cat_map[label_selected]
dataset_selected = DATASETS[dataset_slug]

model_selected = st.sidebar.selectbox("Gen AI model", MODEL_OPTIONS)

prompt_mode_label = st.sidebar.selectbox(
    "Prompt mode",
    list(PROMPT_MODE_LABEL_TO_KEY.keys()),
)
prompt_mode_key = PROMPT_MODE_LABEL_TO_KEY[prompt_mode_label]

# manual override for phone layout
mobile_layout = st.sidebar.checkbox("Mobile layout (2 columns)", value=False)
num_cols = 2 if mobile_layout else 4

# ======================================================
# RESET on LLM model change:
# - revert to ORIGINAL (no random_position)
# - clear highlighted selection
# - show base dataset until user clicks Run Simulator / Shuffle listing positions
# ======================================================
if "prev_model_selected" not in st.session_state:
    st.session_state["prev_model_selected"] = model_selected

if st.session_state["prev_model_selected"] != model_selected:
    st.session_state["prev_model_selected"] = model_selected

    st.session_state["rand_variant"] = 0
    # st.session_state["rand_clicks"] = 0

    st.session_state["agent_sku"] = None
    st.session_state["agent_title"] = None

    st.session_state["df"] = load_base_dataset(dataset_selected)

    st.rerun()


# ------------------------------------------------------
# Prompt text UI
# ------------------------------------------------------
if prompt_mode_key == "custom":
    user_prompt = st.sidebar.text_area(
        "Custom shopping prompt",
        value="",
        height=300,
        disabled=False,
    )
else:
    st.sidebar.text_area(
        "Prompt used (initial part)",
        value=PROMPT_TEXTS[prompt_mode_key],
        height=200,
        disabled=True,
    )
    user_prompt = None

with st.sidebar.expander("ℹ️ See the rest of the prompt", expanded=False):
    st.markdown(
        """
        <p style="font-size:0.8rem; color:#6b7280; margin-top:0;">
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
    base_df = load_base_dataset(dataset_selected)
    st.session_state["df"] = base_df
    st.session_state["agent_sku"] = None
    st.session_state["agent_title"] = None
    st.session_state["prev_dataset_slug"] = dataset_slug

    # Reset random-position state per category
    st.session_state["rand_variant"] = 0
    # st.session_state["rand_clicks"] = 0

if "df" not in st.session_state:
    st.session_state["df"] = load_base_dataset(dataset_selected)

df = st.session_state["df"]

# ======================================================
# RUN EXPERIMENT OR LOAD PRECOMPUTED (sets agent_sku)
# ======================================================
if run_button:
    if prompt_mode_key == "custom":
        # Clear previous experiment logs for this dataset
        dataset_log_dir = EXPERIMENT_LOGS_DIR / dataset_slug
        if dataset_log_dir.exists():
            shutil.rmtree(dataset_log_dir)

        with st.status(
            "Your Agent is Shopping with your custom prompt... It may take up to two minutes to run...",
            expanded=False,
        ):
            retcode = run_aces(dataset_selected, model_selected, prompt=user_prompt)

        if retcode != 0:
            st.error("ACES run failed. Please try again.")
            st.stop()

        csv_path = get_latest_experiment_csv(dataset_slug)
        if not csv_path:
            st.error("No experiment_data.csv found — ACES may not have produced output.")
            st.stop()

        df_run = pd.read_csv(csv_path)

        # For custom mode, do not use random-position toggles
        st.session_state["rand_variant"] = 0
        # st.session_state["rand_clicks"] = 0

    else:
        # Precomputed modes
        # If Default prompt, respect the current rand_variant (0/1/2).
        rv = st.session_state.get("rand_variant", 0) if prompt_mode_key == "default" else 0

        df_run = load_precomputed_results(
            dataset_slug=dataset_slug,
            prompt_mode_key=prompt_mode_key,
            model_selected=model_selected,
            rand_variant=rv,
        )
        if df_run is None or df_run.empty:
            st.stop()

    agent_sku, agent_title = extract_agent_pick(df_run)
    st.session_state["df"] = df_run
    st.session_state["agent_sku"] = agent_sku
    st.session_state["agent_title"] = agent_title
    st.rerun()

# ======================================================
# SORT & AGENT SELECTION
# ======================================================
if "assigned_position" in df.columns:
    df = df.sort_values("assigned_position").reset_index(drop=True)

agent_selected_sku = st.session_state.get("agent_sku", None)

# ======================================================
# GRID VIEW WITH AGENT HIGHLIGHT
# ======================================================
sku_order_current = df["sku"].tolist() if "sku" in df.columns else []
sku_to_row = {row["sku"]: row for _, row in df.iterrows()} if "sku" in df.columns else {}

with st.container(border=True):
    st.markdown(
        '<div class="mock-marketplace-title">Mock marketplace recommendations page</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(num_cols)

    for i, sku in enumerate(sku_order_current):
        if sku not in sku_to_row:
            continue

        row = sku_to_row[sku]
        is_agent_pick = (agent_selected_sku is not None and sku == agent_selected_sku)

        with cols[i % num_cols]:
            # IMAGE
            render_product_image(row.get("image_url"), highlight=is_agent_pick)

            # WHITE CONTENT
            st.markdown(
                '<div style="background-color:#ffffff; padding:8px 6px 10px 6px; border-radius:12px;">',
                unsafe_allow_html=True,
            )

            full_title = str(row.get("title", ""))
            title_color = "#dc2626" if is_agent_pick else "#111827"
            st.markdown(
                f"""
                <div class="product-title" style="color:{title_color};">
                    {full_title}
                </div>
                """,
                unsafe_allow_html=True,
            )

            rating = row.get("rating", None)
            rating_count = row.get("rating_count", None)
            if rating is not None and pd.notna(rating):
                stars = rating_to_stars(rating)
                reviews_part = ""
                if rating_count is not None and pd.notna(rating_count):
                    try:
                        reviews_part = f" ({int(rating_count)})"
                    except Exception:
                        reviews_part = f" ({rating_count})"
                st.markdown(
                    f'<p class="product-meta">{stars} {float(rating):.1f}{reviews_part}</p>',
                    unsafe_allow_html=True,
                )

            price = row.get("price", None)
            if price is not None and pd.notna(price):
                st.markdown(
                    f'<p class="product-meta"><strong>${float(price):.2f}</strong></p>',
                    unsafe_allow_html=True,
                )

            low_stock = bool(row.get("low_stock", False))
            stock_quantity = row.get("stock_quantity", None)
            overall_pick = bool(row.get("overall_pick", False))
            sponsored = bool(row.get("sponsored", False))

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
                    <span style="
                        font-size:0.8rem;
                        font-weight:500;
                        color:#111827;
                        background-color:#facc15;
                        border:1px solid #facc15;
                        padding:4px 8px;
                        border-radius:4px;
                        display:inline-block;
                    ">
                        Add to Cart
                    </span>
                    {pill_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("</div>", unsafe_allow_html=True)  # close white body

# ======================================================
# CHANGE POSITION (LAST BUTTON ON PAGE)
# Put this at the VERY END of the script (after the grid)
# ======================================================

# Only Default prompt can change position; otherwise reset and disable
if prompt_mode_key != "default":
    st.session_state["rand_variant"] = 0
    # st.session_state["rand_clicks"] = 0

change_pos_disabled = (prompt_mode_key != "default")

change_pos_clicked = st.sidebar.button(
    "🔀 Shuffle listing positions",
    disabled=change_pos_disabled,
    help="Click Shuffle listing positions to explore how listing positions affect the agent's choice.",
    key="change_position_last",  
)

# Optional status text (not a button)
if prompt_mode_key == "default":
    label = "Original" if st.session_state["rand_variant"] == 0 else f"Random {st.session_state['rand_variant']}"
    st.sidebar.caption(f"Position mode: {label}")

if change_pos_clicked and not change_pos_disabled:
    st.session_state["rand_variant"] = (st.session_state.get("rand_variant", 0) + 1) % 3

    # IMPORTANT: do NOT auto-run / do NOT highlight
    st.session_state["agent_sku"] = None
    st.session_state["agent_title"] = None

    # Update the displayed listing order immediately
    if st.session_state["rand_variant"] == 0:
        # back to original positions (base dataset)
        st.session_state["df"] = load_base_dataset(dataset_selected)
    else:
        # show shuffled positions (precomputed listing), but still no highlight until Run Simulator
        df_listing = load_precomputed_results(
            dataset_slug=dataset_slug,
            prompt_mode_key="default",
            model_selected=model_selected,
            rand_variant=st.session_state["rand_variant"],
        )
        if df_listing is None or df_listing.empty:
            st.stop()
        st.session_state["df"] = df_listing

    st.rerun()
