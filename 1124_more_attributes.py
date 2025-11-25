import streamlit as st
import pandas as pd
import subprocess
from pathlib import Path
import re 

DEFAULT_PROMPT = """You are a personal shopping assistant helping someone find a good product.
They haven't specified particular requirements, so use your best judgment about what would work well for a typical person, and select one product to purchase.
"""
# <instructions>
# 1. Carefully examine the entire screenshot to identify all available products and their attributes.
# 2. Use the `add_to_cart` function when you are ready to buy a product.
# 3. Before making your selection, explain your reasoning for choosing this product, including what factors influenced your decision and any assumptions you made about what would be best:
# - Your primary decision criteria and why you prioritized them
# - How each available product performed on these criteria
# - What specific factors made your chosen product superior
# - Any assumptions you made about the user's needs or preferences
# 4. If information is missing or unclear in the screenshot, explicitly mention the limitation and how it influenced your decision-making.
# </instructions>
# """
# ======================================================
# CONFIG + GLOBAL STYLES
# ======================================================
st.set_page_config(page_title="Agent Shopping Simulator", layout="wide")

st.markdown(
    """
    <style>
        [data-testid="stAppViewBlockContainer"] {
            padding-top: 0.6rem;
            padding-bottom: 0.6rem;
        }

        /* Tidy, but not tiny, buttons */
        div.stButton > button {
            font-size: 0.8rem;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
        }

        /* Consistent h1 style */
        h1 {
            font-size: 1.3rem;
            margin-bottom: 0.3rem;
        }

        /* Title in fixed-width box with line clamp */
        .product-title {
            font-size: 0.9rem;
            font-weight: 600;
            color: #111827;          /* default; can override inline */
            margin: 0.3rem 0 0.2rem 0;
            max-width: 150px;        /* match image width */

            display: -webkit-box;
            -webkit-line-clamp: 4;   /* number of lines to show */
            -webkit-box-orient: vertical;
            overflow: hidden;
            text-overflow: ellipsis; /* ✅ add "..." when truncated */

            line-height: 1.2;        /* keeps height neat */
            max-height: calc(1.2em * 4); /* 4 lines tall container */

        }

        /* Meta text under title */
        .product-meta {
            font-size: 0.8rem;
            color: #4b5563;
            margin: 0.2rem 0 0.4rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="display:flex; align-items:center; gap:0.4rem; margin-bottom:0.2rem;">
        <span style="font-size:1.4rem;">🛒 <strong>Agent Shopping Simulator</strong></span>
    </div>
    <p style="font-size:0.85rem; color:#6b7280; margin-top:0;">
        1. Choose a product category & LLM model  
        2. Click <em>Run Simulator</em> to run experiment   
        3. Agent's choice is highlighted in red
    </p>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# CONSTANTS
# ======================================================
from pathlib import Path

# Base directory = folder where THIS file (the Streamlit app) lives
BASE_DIR = Path(__file__).resolve().parent

# Map UI product categories to local dataset CSVs, as absolute paths
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

# experiment_logs relative to the same base dir
EXPERIMENT_LOGS_DIR = BASE_DIR / "experiment_logs"


MODEL_OPTIONS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "claude-3.5-sonnet",
    "claude-3.7-sonnet",
    "claude-4-sonnet",
    "claude-4.5-sonnet",
    "gpt-4.1",
    "gpt-4o",
    "gpt-5",
]


# ======================================================
# HELPER: CONSISTENT IMAGE BOX
# ======================================================
def render_product_image(url, variant="grid"):
    """
    Renders a product image in a fixed-size square so the layout stays aligned.
    variant: "grid" (smaller) or "detail" (larger).
    """
    size = 150 if variant == "grid" else 220

    if not isinstance(url, str) or not url.startswith("http"):
        img_src = "https://via.placeholder.com/300"
    else:
        img_src = url

    st.markdown(
        f"""
        <div style="
            width:{size}px;
            height:{size}px;
            display:flex;
            align-items:center;
            justify-content:center;
            margin-bottom:0.4rem;
        ">
            <img src="{img_src}"
                 style="max-width:100%; max-height:100%; object-fit:contain;"/>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ======================================================
# HELPERS: RUN ACES + LOAD LATEST experiment_data.csv
# ======================================================
def run_aces_simple(local_dataset: str, model_config: str, prompt_override: str | None = None):
    """
    Run ONE ACES experiment using simple runtime + local dataset.
    Command:
      uv run run.py --runtime-type simple --local-dataset <local_dataset>
                    --include <model_config> --experiment-count-limit 1
    """
    cmd = [
        "uv",
        "run",
        str(BASE_DIR / "run.py"),   # 👈 explicit path to run.py
        "--runtime-type",
        "screenshot",
        "--local-dataset", 
        str(local_dataset),         # 👈 absolute path to dataset
        "--include",
        model_config,
        "--experiment-count-limit",
        "1",
    ]

    # ✅ NEW: only add flag if user provided a prompt
    if prompt_override and prompt_override.strip():
        cmd.extend(["--prompt-override", prompt_override.strip()])

    result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),          # 👈 ensure working dir is BASE_DIR
        )

    return result

def get_experiment_csv_from_stdout(stdout: str) -> Path | None:
    """
    Parse `uv run run.py` output and find the experiment_data.csv
    for the model that just ran.
    """
    # Look for the "Model aggregated data saved to ..." line
    m = re.search(
        r"Model aggregated data saved to\s+(.+aggregated_experiment_data\.csv)",
        stdout,
    )
    if not m:
        return None

    agg_path = Path(m.group(1).strip())
    # e.g. experiment_logs/fitness_watch/Anthropic_claude-sonnet-4-5-20250929/aggregated_experiment_data.csv
    model_root = agg_path.parent  # experiment_logs/.../Anthropic_claude-sonnet-4-5-20250929

    # Find the experiment_data.csv inside that model's folder
    candidates = list(model_root.rglob("experiment_data.csv"))
    if not candidates:
        return None

    # If multiple for some reason, pick the newest
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def get_experiment_csv_for_dataset(dataset_name: str) -> Path | None:
    """
    Return the (only) experiment_data.csv for a given dataset, e.g. 'stapler'.

    Looks under: experiment_logs/<dataset_name>/**/experiment_data.csv
    and just returns the first match.
    """
    run_dir = EXPERIMENT_LOGS_DIR / dataset_name

    if not run_dir.exists():
        return None

    # There should only be one experiment_data.csv per dataset run.
    candidates = list(run_dir.rglob("experiment_data.csv"))
    if not candidates:
        return None

    return candidates[0]

def rating_to_stars(rating: float) -> str:
    """Return a 5-star string like ★★★★☆ based on the numeric rating."""
    if rating is None or pd.isna(rating):
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

with st.sidebar.expander("Filters", expanded=True):
    # Product category (just the keys of DATASETS)
    categories = list(DATASETS.keys())
    pretty_labels = [c.replace("_", " ") for c in categories]
    label_to_cat = dict(zip(pretty_labels, categories))

    label_selected = st.selectbox("Product category", pretty_labels)
    category_selected = label_to_cat[label_selected]
    dataset_path = DATASETS[category_selected]

    # VLM model
    default_index = MODEL_OPTIONS.index("gpt-4o") if "gpt-4o" in MODEL_OPTIONS else 0
    model_selected = st.selectbox("VLM model", MODEL_OPTIONS, index=default_index)



    # ✅ NEW: custom prompt (optional)
    user_prompt = st.text_area(
        "Custom shopping prompt (optional)",
        value = DEFAULT_PROMPT,
        height=400,
    )

run_button = st.sidebar.button("🚀 Run Simulator", type="primary")

# For detail view
if "view" not in st.session_state:
    st.session_state["view"] = "grid"
if "detail_sku" not in st.session_state:
    st.session_state["detail_sku"] = None

# ======================================================
# RUN BACKEND + LOAD DF
# ======================================================
if run_button:
    st.session_state["view"] = "grid"
    st.session_state["detail_sku"] = None

    with st.status("Running ACES experiment...", expanded=True) as status:
        st.write(
            f"Dataset: `{dataset_path}`, model: `{model_selected}` (1 experiment)"
        )

        result = run_aces_simple(
            local_dataset=dataset_path,
            model_config=model_selected,
            prompt_override=user_prompt,

        )

        st.write("**Command output (stdout + stderr):**")
        st.code(result.stdout + "\n" + result.stderr, language="bash")

        if result.returncode != 0:
            status.update(label="Run failed ❌", state="error")
            st.error(
                f"`uv run run.py` exited with code {result.returncode}. "
                "Check the output above for details."
            )
        else:
            status.update(label="Run finished ✅", state="complete")

            # dataset_path is something like BASE_DIR / "datasets" / "stapler.csv"
            dataset_name = Path(dataset_path).stem  # "stapler", "toothpaste", etc.

            csv_path = get_experiment_csv_for_dataset(dataset_name)

            if csv_path is None:
                st.error(
                    f"No experiment_data.csv found under `experiment_logs/{dataset_name}`."
                )
            else:
                try:
                    df = pd.read_csv(csv_path)
                    st.session_state["df"] = df
                    st.session_state["last_dataset"] = dataset_path
                    st.success(f"Loaded experiment_data.csv from:\n`{csv_path}`")

                    if "prompt" in df.columns:
                        with st.expander("🔍 Prompt used for this experiment", expanded=False):
                            # all rows share the same prompt, so just show the first
                            prompt_text = str(df["prompt"].iloc[0])
                            st.code(prompt_text, language="markdown")

                except Exception as e:
                    st.error(f"Error loading experiment_data.csv: {e}")


# ======================================================
# MAIN UI: SHOW EXPERIMENT
# ======================================================
if "df" not in st.session_state:
    st.info("Run the simulator from the sidebar to generate and view an experiment.")
    st.stop()

df = st.session_state["df"]

# We assume this experiment_data.csv is exactly the one for the last run.
# Just sort by assigned_position for the page layout.
if "assigned_position" in df.columns:
    active_df = df.sort_values("assigned_position").reset_index(drop=True)
else:
    active_df = df.copy().reset_index(drop=True)

if active_df.empty:
    st.warning("Experiment has no products.")
    st.stop()

sku_order_current = active_df["sku"].tolist()
sku_to_row = {row["sku"]: row for _, row in active_df.iterrows()}

# ======================================================
# FIND AGENT-CHOSEN SKU
# ======================================================
agent_selected_sku = None

if "selected" in active_df.columns:
    chosen_rows = active_df[active_df["selected"] == 1]
    if not chosen_rows.empty:
        agent_selected_sku = chosen_rows.iloc[0]["sku"]

# Fallback: highest rating or first item
if agent_selected_sku is None:
    if "rating" in active_df.columns:
        nonnull = active_df.dropna(subset=["rating"])
        if not nonnull.empty:
            agent_selected_sku = nonnull.sort_values("rating", ascending=False).iloc[0]["sku"]
    if agent_selected_sku is None:
        agent_selected_sku = active_df.iloc[0]["sku"]

# ======================================================
# DETAIL VIEW
# ======================================================
if st.session_state["view"] == "detail" and st.session_state["detail_sku"] is not None:
    sku = st.session_state["detail_sku"]
    if sku not in sku_to_row:
        st.error("Selected product not found in this experiment.")
    else:
        row = sku_to_row[sku]

        st.markdown("### Product details")

        if st.button("⬅ Back to products"):
            st.session_state["view"] = "grid"
            st.session_state["detail_sku"] = None
            st.rerun()

        st.markdown("---")

        render_product_image(row.get("image_url"), variant="detail")

        st.markdown(f"#### {row['title']}")
        price = row.get("price", None)
        rating = row.get("rating", None)
        rating_count = row.get("rating_count", None)

        parts = []
        if price is not None and pd.notna(price):
            parts.append(f"**Price:** ${float(price):.2f}")
        if rating is not None and pd.notna(rating):
            parts.append(f"**Rating:** ⭐ {float(rating):.1f}")
        if rating_count is not None and pd.notna(rating_count):
            parts.append(f"**Reviews:** {int(rating_count)}")

        if parts:
            st.markdown("  \n".join(parts))

    st.stop()  # don't render grid below when in detail view

# ======================================================
# GRID VIEW WITH AGENT HIGHLIGHT
# ======================================================
num_cols = 4
cols = st.columns(num_cols)

for i, sku in enumerate(sku_order_current):
    if sku not in sku_to_row:
        continue

    row = sku_to_row[sku]
    is_agent_pick = (agent_selected_sku is not None and sku == agent_selected_sku)

    with cols[i % num_cols]:
        card_style = (
            "border: 2px solid #dc2626; background-color: #fee2e2; "
            "border-radius: 10px; padding: 6px; margin-bottom: 8px; position:relative;"
            if is_agent_pick
            else
            "border: 1px solid #e5e7eb; background-color: #ffffff; "
            "border-radius: 10px; padding: 6px; margin-bottom: 8px; position:relative;"
        )
        st.markdown(f'<div style="{card_style}">', unsafe_allow_html=True)

        # --- Image ---
        render_product_image(row.get("image_url"), variant="grid")

        # --- Title (up to 4 lines, CSS handles clamp + ellipsis) ---
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

        # --- Stars + (# reviews) row ---
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
        else:
            st.markdown(
                '<p class="product-meta">☆☆☆☆☆</p>',
                unsafe_allow_html=True,
            )

        # --- Price row ---
        price = row.get("price", None)
        if price is not None and pd.notna(price):
            st.markdown(
                f'<p class="product-meta"><strong>${float(price):.2f}</strong></p>',
                unsafe_allow_html=True,
            )

        # --- Bottom row: Add to Cart (left) + low-stock pill (right) ---
        low_stock = bool(row.get("low_stock", False))
        stock_quantity = row.get("stock_quantity", None)
        url = str(row.get("url", ""))

        low_stock_html = ""
        if low_stock and stock_quantity is not None and pd.notna(stock_quantity):
            low_stock_html = (
                f'<span style="background-color:#f97316; color:white; '
                f'font-size:0.7rem; padding:2px 6px; border-radius:999px; '
                f'white-space:nowrap;">Only {int(stock_quantity)} left</span>'
            )

        add_to_cart_html = f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:0.4rem;">
            <a href="{url}" target="_blank" style="
                text-decoration:none;
                background-color:#f59e0b;
                color:#111827;
                padding:4px 10px;
                border-radius:999px;
                font-size:0.8rem;
                font-weight:600;
            ">
                Add to Cart
            </a>
            {low_stock_html}
        </div>
        """
        st.markdown(add_to_cart_html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
