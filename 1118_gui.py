import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import gdown
import os

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
        1. Choose a product category & LLM model from filter 
        2. Reorder items using <em>Change position</em> · 
        3. Agent's choice is highlighted in red
    </p>
    """,
    unsafe_allow_html=True,
)

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
# DATA CONFIG
# ======================================================
FILE_ID = "1N1uMzSz4WKDELD0FkMLXrjiXo5yNZeG3"
CSV_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
LOCAL_PATH = "concat_df_downloaded.csv"

# ======================================================
# LOAD DATA (silent)
# ======================================================
@st.cache_data
def load_data():
    if not os.path.exists(LOCAL_PATH):
        gdown.download(CSV_URL, LOCAL_PATH, quiet=True)
    try:
        df = pd.read_csv(LOCAL_PATH)
        return df
    except Exception as e:
        st.error(f"Failed to read CSV file: {e}")
        st.stop()

df = load_data()
if df.empty:
    st.error("Data not loaded. Check CSV source or file format.")
    st.stop()

# ======================================================
# SIDEBAR FILTERS
# ======================================================
st.sidebar.title("Controls")

with st.sidebar.expander("Filters", expanded=True):
    # ---- Product category: pretty labels without underscores ----
    query_values = sorted(df["query"].dropna().unique().tolist())
    query_labels = [q.replace("_", " ") for q in query_values]
    label_to_query = dict(zip(query_labels, query_values))

    query_label_selected = st.selectbox("Product category", query_labels)
    query_selected = label_to_query[query_label_selected]

    # ---- VLM model as-is ----
    models = sorted(df["model_name_dir"].dropna().unique().tolist())
    model_selected = st.selectbox("VLM model", models)

# ======================================================
# FILTER DATA FOR THIS CONDITION
# ======================================================
subset = df[
    (df["query"] == query_selected)
    & (df["model_name_dir"] == model_selected)
    & (df["prompt_type"] == "default")
].copy()

if subset.empty:
    st.warning("No rows found for this combination.")
    st.stop()

# All experiments available for this (query, model)
all_exps = sorted(subset["experiment_number"].unique())
initial_exp = all_exps[0]

# Helper to get layout df for a given experiment
def get_experiment_layout(exp_num: int) -> pd.DataFrame:
    exp_df = subset[subset["experiment_number"] == exp_num].copy()
    return exp_df.sort_values("assigned_position").reset_index(drop=True)

# Default layout = first experiment
base_exp_df = get_experiment_layout(initial_exp)
sku_order_default = base_exp_df["sku"].tolist()

# ======================================================
# SESSION STATE INIT
# ======================================================
combo_key = (query_selected, model_selected)

if "combo_key" not in st.session_state or st.session_state["combo_key"] != combo_key:
    st.session_state["combo_key"] = combo_key
    st.session_state["current_exp"] = initial_exp
    st.session_state["selected_sku"] = sku_order_default[0]
    st.session_state["view"] = "grid"
    st.session_state["detail_sku"] = None

if "current_exp" not in st.session_state:
    st.session_state["current_exp"] = initial_exp

if "selected_sku" not in st.session_state:
    st.session_state["selected_sku"] = sku_order_default[0]

if "view" not in st.session_state:
    st.session_state["view"] = "grid"

if "detail_sku" not in st.session_state:
    st.session_state["detail_sku"] = None

current_exp = st.session_state["current_exp"]
selected_sku = st.session_state["selected_sku"]

# ======================================================
# ACTIVE LAYOUT FROM CURRENT EXPERIMENT
# ======================================================
active_df = get_experiment_layout(current_exp)

# If selected_sku somehow not in this experiment, reset to first
if selected_sku not in active_df["sku"].tolist():
    selected_sku = active_df["sku"].iloc[0]
    st.session_state["selected_sku"] = selected_sku

sku_order_current = active_df["sku"].tolist()

# Build SKU → row mapping from current experiment layout
sku_to_row = {row["sku"]: row for _, row in active_df.iterrows()}

# ======================================================
# SIDEBAR: REORDER CONTROLS (DRIVES WHICH EXPERIMENT WE JUMP TO)
# ======================================================
st.sidebar.markdown("---")
st.sidebar.subheader("Layout control")

current_index = sku_order_current.index(selected_sku)

# Use current experiment's row for sidebar title
sidebar_row = sku_to_row[selected_sku]
st.sidebar.markdown("**Product to move:**")
st.sidebar.caption(sidebar_row["title"])
# run experiments:
# st.sidebar.write(f"Current position: **{current_index + 1}** (in experiment {current_exp})")
st.sidebar.write(f"Current position: **{current_index + 1}**")


new_position = st.sidebar.slider(
    "Move to position",
    min_value=1,
    max_value=len(sku_order_current),
    value=current_index + 1,
    step=1,
    key="new_position",
)

col_apply, col_reset = st.sidebar.columns(2)

with col_apply:
    if st.button("Apply"):
        # Target position index (0-based) we want this SKU to occupy
        target_pos = new_position - 1

        # Find all rows where this sku is at that assigned_position
        candidates = subset[
            (subset["sku"] == selected_sku) &
            (subset["assigned_position"] == target_pos)
        ]

        if not candidates.empty:
            # Take the earliest experiment_number for this (sku, position)
            new_exp = int(candidates["experiment_number"].min())
            st.session_state["current_exp"] = new_exp
            st.success(
                f"Jumped to experiment {new_exp} "
                f"(where {selected_sku} is at position {target_pos})."
            )
        else:
            st.warning(
                f"No experiment found where {selected_sku} is at position {target_pos}. "
                f"Staying on experiment {current_exp}."
            )

        st.rerun()

with col_reset:
    if st.button("Reset"):
        st.session_state["current_exp"] = initial_exp
        st.session_state["selected_sku"] = sku_order_default[0]
        st.session_state["view"] = "grid"
        st.session_state["detail_sku"] = None
        st.info(f"Reset to experiment {initial_exp}.")
        st.rerun()

# ======================================================
# AGENT-CHOSEN SKU (ALWAYS ONE) FROM CURRENT EXPERIMENT
# ======================================================
agent_selected_sku = None

if not active_df.empty:
    if "selected" in active_df.columns:
        chosen_rows = active_df[active_df["selected"] == 1]
        if not chosen_rows.empty:
            agent_selected_sku = chosen_rows.iloc[0]["sku"]

    # Fallback: highest rating
    if agent_selected_sku is None:
        if "rating" in active_df.columns:
            nonnull = active_df.dropna(subset=["rating"])
            if not nonnull.empty:
                agent_selected_sku = nonnull.sort_values("rating", ascending=False).iloc[0]["sku"]
        # Final fallback: first row
        if agent_selected_sku is None:
            agent_selected_sku = active_df.iloc[0]["sku"]

# ======================================================
# DETAIL VIEW (when a title is clicked)
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
            st.rerun()

        st.markdown("---")

        # Image in larger, fixed-size box
        render_product_image(row.get("image_url"), variant="detail")

        st.markdown(f"#### {row['title']}")
        st.markdown(
            f"""
            **Price:** ${row['price']:.2f}  
            **Rating:** ⭐ {row['rating']:.1f}  
            **Reviews:** {int(row['rating_count'])}  
            """,
        )

    st.rerun()  # do not render the grid below when in detail view

# ======================================================
# PAGE PREVIEW WITH CLICKABLE TITLES + AGENT HIGHLIGHT
# ======================================================
num_cols = 4
cols = st.columns(num_cols)

for i, sku in enumerate(sku_order_current):
    if sku not in sku_to_row:
        continue

    row = sku_to_row[sku]
    is_agent_pick = (agent_selected_sku is not None and sku == agent_selected_sku)

    with cols[i % num_cols]:
        # Card wrapper, highlight agent's pick
        card_style = (
            "border: 2px solid #dc2626; background-color: #fee2e2; "
            "border-radius: 10px; padding: 6px; margin-bottom: 8px;"
            if is_agent_pick
            else
            "border: 1px solid #e5e7eb; background-color: #ffffff; "
            "border-radius: 10px; padding: 6px; margin-bottom: 8px;"
        )
        st.markdown(f'<div style="{card_style}">', unsafe_allow_html=True)

        # Image in fixed-size box
        render_product_image(row.get("image_url"), variant="grid")

        # Title: full text, clamped to 2 lines, red if agent pick
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

        # Button to open detail view
        if st.button("View details", key=f"view_{sku}"):
            st.session_state["detail_sku"] = sku
            st.session_state["view"] = "detail"
            st.rerun()

        # Meta line (price, rating, reviews)
        st.markdown(
            f"""
            <p class="product-meta">
                ${row['price']:.2f} · ⭐ {row['rating']:.1f} · {int(row['rating_count'])} reviews
            </p>
            """,
            unsafe_allow_html=True,
        )

        # Button to select product for moving (layout control)
        if st.button("Change position", key=f"select_{sku}"):
            st.session_state["selected_sku"] = sku
            st.rerun()

        # Close card wrapper
        st.markdown("</div>", unsafe_allow_html=True)