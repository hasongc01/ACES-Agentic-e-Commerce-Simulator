import streamlit as st
import pandas as pd
import subprocess
from pathlib import Path

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
            -webkit-line-clamp: 2;   /* number of lines to show */
            -webkit-box-orient: vertical;
            overflow: hidden;
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
        2. Click <em>Run Simulator</em> to run 1 experiment on the local dataset  
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

EXPERIMENT_LOGS_DIR = Path("experiment_logs")

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
def run_aces_screenshot(local_dataset: str, model_config: str):
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
        # "simple",
        "screenshot",
        "--local-dataset", 
        str(local_dataset),         # 👈 absolute path to dataset
        "--include",
        model_config,
        "--experiment-count-limit",
        "1",
    ]
    result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),          # 👈 ensure working dir is BASE_DIR
        )

    return result


def find_latest_experiment_csv() -> Path | None:
    """
    Find the most recent experiment_data.csv anywhere under experiment_logs/.
    This assumes each run creates a new or updated master_experiment_*/experiment_data.csv.
    """
    if not EXPERIMENT_LOGS_DIR.exists():
        return None

    candidates = list(EXPERIMENT_LOGS_DIR.rglob("experiment_data.csv"))
    if not candidates:
        return None

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest

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

        result = run_aces_screenshot(
            local_dataset=dataset_path,
            model_config=model_selected,
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

            csv_path = find_latest_experiment_csv()
            if csv_path is None:
                st.error("No experiment_data.csv found under `experiment_logs/`.")
            else:
                try:
                    df = pd.read_csv(csv_path)
                    st.session_state["df"] = df
                    st.session_state["last_dataset"] = dataset_path
                    st.success(f"Loaded experiment_data.csv from:\n`{csv_path}`")
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
            "border-radius: 10px; padding: 6px; margin-bottom: 8px;"
            if is_agent_pick
            else
            "border: 1px solid #e5e7eb; background-color: #ffffff; "
            "border-radius: 10px; padding: 6px; margin-bottom: 8px;"
        )
        st.markdown(f'<div style="{card_style}">', unsafe_allow_html=True)

        render_product_image(row.get("image_url"), variant="grid")

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

        if st.button("View details", key=f"view_{sku}"):
            st.session_state["detail_sku"] = sku
            st.session_state["view"] = "detail"
            st.rerun()

        price = row.get("price", None)
        rating = row.get("rating", None)
        rating_count = row.get("rating_count", None)

        meta_parts = []
        if price is not None and pd.notna(price):
            meta_parts.append(f"${float(price):.2f}")
        if rating is not None and pd.notna(rating):
            meta_parts.append(f"⭐ {float(rating):.1f}")
        if rating_count is not None and pd.notna(rating_count):
            meta_parts.append(f"{int(rating_count)} reviews")

        if meta_parts:
            st.markdown(
                f'<p class="product-meta">{" · ".join(meta_parts)}</p>',
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)
