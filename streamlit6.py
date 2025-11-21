import streamlit as st
import subprocess
import os
import glob
from pathlib import Path
from PIL import Image
import pandas as pd

# ===========================
# CONFIG
# ===========================
st.set_page_config(page_title="Agentic AI Shopping Simulator", layout="wide")

# ---------- PATH SETUP (robust for local + cloud) ----------
ROOT_DIR = Path(__file__).resolve().parent  # folder where this file lives
DATASET_ROOT = ROOT_DIR / "datasets"
EXPERIMENT_LOGS_ROOT = ROOT_DIR / "experiment_logs"

# ---------- TOP-LEVEL DEBUG INFO ----------
st.write("### DEBUG: Environment info")
st.write("CWD:", os.getcwd())
st.write("__file__:", __file__)
st.write("ROOT_DIR:", str(ROOT_DIR))

st.write("DATASET_ROOT:", str(DATASET_ROOT), "exists:", DATASET_ROOT.exists())
if DATASET_ROOT.exists():
    try:
        st.write("DATASET_ROOT contents:", [p.name for p in DATASET_ROOT.iterdir()])
    except Exception as e:
        st.write("Could not list DATASET_ROOT contents:", e)

st.write(
    "EXPERIMENT_LOGS_ROOT:", str(EXPERIMENT_LOGS_ROOT),
    "exists:", EXPERIMENT_LOGS_ROOT.exists()
)
if EXPERIMENT_LOGS_ROOT.exists():
    try:
        subdirs = [p.name for p in EXPERIMENT_LOGS_ROOT.iterdir() if p.is_dir()]
        st.write("EXPERIMENT_LOGS_ROOT subdirs:", subdirs)
    except Exception as e:
        st.write("Could not list EXPERIMENT_LOGS_ROOT contents:", e)

st.markdown("---")

st.markdown(
    """
    <h1>🛒 Agentic AI Shopping Simulator (DEBUG MODE)</h1>
    <p class="small-muted">
        1. Choose a product category, model, and prompt on the left. <br/>
        2. Click <strong>Run Simulation</strong>. <br/>
        3. This version prints detailed debug info about datasets, subprocess, and experiment logs.
    </p>
    """,
    unsafe_allow_html=True,
)

DEFAULT_PROMPT = (
    "You are a personal shopping assistant helping someone find a good product. "
    "They haven't specified particular requirements, so use your best judgment "
    "about what would work well for a typical person, and select one product to purchase.\n\n"
    "<instructions>\n"
    "1. Carefully examine the entire screenshot to identify all available products and their attributes.\n"
    "2. Use the add_to_cart function when you are ready to buy a product.\n"
    "3. Before making your selection, explain your reasoning for choosing this product, including what factors "
    "influenced your decision and any assumptions you made about what would be best.\n"
    "4. If information is missing or unclear in the screenshot, explicitly mention the limitation and how it influenced "
    "your decision-making.\n"
    "</instructions>"
)

LOCAL_DATA_FLAG = "--local-dataset"  # correct CLI flag
MAX_EXPERIMENTS_PER_COMBO = 500      # per (category, model)

# ===========================
# SESSION STATE
# ===========================
defaults = {
    "last_screenshot_path": None,
    "last_journey_gif": None,
    "last_result_error": None,
    "has_run": False,
    "product_title": None,
    "image_url": None,
    "url": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# store experiment counts per (category, model)
if "experiment_counts" not in st.session_state:
    # key: "category::model" -> int
    st.session_state["experiment_counts"] = {}

# ===========================
# LAYOUT
# ===========================
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("Filters & Prompt")

    backend_categories = [
        "fitness_watch",
        "iphone_16_pro_cover",
        "mousepad",
        "stapler",
        "toilet_paper",
        "tooth_paste",
        "usb_cable",
        "washing_machine",
    ]

    category_labels = [c.replace("_", " ").title() for c in backend_categories]
    label_to_category = dict(zip(category_labels, backend_categories))

    category_label = st.selectbox("Select a product category:", category_labels)
    category = label_to_category[category_label]  # e.g. "washing_machine"

    model = st.selectbox(
        "Choose a model:",
        [
            "gemini-2.0-flash",
            "gemini-2.5-flash",
            "claude-3.5-sonnet",
            "claude-3.7-sonnet",
            "claude-4-sonnet",
            "claude-4.5-sonnet",
            "gpt-4.1",
            "gpt-4o",
            "gpt-5",
        ],
    )

    prompt = st.text_area(
        "Enter your shopping prompt:",
        value=DEFAULT_PROMPT,
        height=160,
    )

    run_button = st.button("🚀 Run Simulation", use_container_width=True)

# ===========================
# RUN SIMULATION
# ===========================
if run_button:
    st.write("### DEBUG: Run button clicked")
    st.write("Selected category:", category)
    st.write("Selected model:", model)

    if not prompt.strip():
        st.warning("Please enter a prompt before running the simulation.")
    else:
        st.session_state["has_run"] = True

        # Clear previous results
        st.session_state["last_result_error"] = None
        st.session_state["last_screenshot_path"] = None
        st.session_state["last_journey_gif"] = None
        st.session_state["product_title"] = None
        st.session_state["image_url"] = None
        st.session_state["url"] = None

        # ---- Dataset path: CSV per category, e.g. datasets/washing_machine.csv ----
        folder_base = category.lower().replace(" ", "_")  # e.g. "washing_machine"
        data_path = DATASET_ROOT / f"{folder_base}.csv"

        st.write("DEBUG: data_path resolved to:", str(data_path))
        st.write("DEBUG: data_path exists:", data_path.exists())

        if not data_path.exists():
            st.session_state["last_result_error"] = (
                f"Local dataset path does not exist on disk: {data_path}"
            )
            st.error(st.session_state["last_result_error"])
        else:
            # track experiments per (category, model)
            combo_key = f"{category}::{model}"
            prev_count = st.session_state["experiment_counts"].get(combo_key, 0)
            new_count = prev_count + 1
            reset_happened = False

            if new_count > MAX_EXPERIMENTS_PER_COMBO:
                new_count = 1
                reset_happened = True

            st.session_state["experiment_counts"][combo_key] = new_count
            st.write("DEBUG: experiment_counts:", st.session_state["experiment_counts"])

            # -------- Run ACES with spinner (Simple runtime, local dataset) --------
            with right_col:
                with st.spinner("🤖 I am still shopping ... (debug: running uv run)"):
                    cmd = (
                        f"uv run run.py "
                        f"--runtime-type simple "
                        f"{LOCAL_DATA_FLAG} \"{str(data_path)}\" "
                        f"--include {model} "
                        f"--experiment-count-limit {new_count}"
                    )

                    st.write("DEBUG: Command to run:")
                    st.code(cmd)

                    # Capture stdout/stderr for debugging
                    result = subprocess.run(
                        cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                    )

                    st.write("DEBUG: subprocess returncode:", result.returncode)
                    if result.stdout:
                        st.write("DEBUG: subprocess STDOUT:")
                        st.code(result.stdout)
                    else:
                        st.write("DEBUG: subprocess STDOUT is empty")

                    if result.stderr:
                        st.write("DEBUG: subprocess STDERR:")
                        st.code(result.stderr)
                    else:
                        st.write("DEBUG: subprocess STDERR is empty")

                if reset_happened:
                    st.info(
                        f"Reached {MAX_EXPERIMENTS_PER_COMBO} experiments for "
                        f"{category_label} / {model}. Counter reset; "
                        f"current run uses experiment-count-limit = {new_count}."
                    )

            # -------- Locate experiment_logs/<category>/.../experiment_data.csv --------
            base_dir = EXPERIMENT_LOGS_ROOT
            run_dir = base_dir / category

            st.write("DEBUG: Looking for experiment logs")
            st.write("DEBUG: base_dir (EXPERIMENT_LOGS_ROOT):", str(base_dir))
            st.write("DEBUG: run_dir (per category):", str(run_dir))
            st.write("DEBUG: base_dir exists:", base_dir.exists())
            st.write("DEBUG: run_dir exists:", run_dir.exists())

            if not base_dir.exists():
                st.session_state["last_result_error"] = "No experiment_logs directory found."
                st.error(st.session_state["last_result_error"])
            elif not run_dir.exists() or not run_dir.is_dir():
                st.session_state["last_result_error"] = (
                    f"No run directory found for category '{category}' under experiment_logs."
                )
                st.error(st.session_state["last_result_error"])
            else:
                try:
                    # Using Path.rglob instead of glob + os.path
                    exp_csv_files = list(run_dir.rglob("experiment_data.csv"))
                    st.write(
                        "DEBUG: Found experiment_data.csv files:",
                        [str(p) for p in exp_csv_files],
                    )
                except Exception as e:
                    st.write("DEBUG: Error while searching for experiment_data.csv:", e)
                    exp_csv_files = []

                if not exp_csv_files:
                    st.session_state["last_result_error"] = (
                        "No experiment_data.csv files found for this category."
                    )
                    st.error(st.session_state["last_result_error"])
                else:
                    # Sort by modification time
                    exp_csv_files_sorted = sorted(
                        exp_csv_files,
                        key=lambda p: p.stat().st_mtime
                    )
                    latest_csv_path = exp_csv_files_sorted[-1]
                    st.write("DEBUG: Using latest experiment_data.csv:", str(latest_csv_path))

                    experiment_folder = latest_csv_path.parent
                    st.write("DEBUG: experiment_folder:", str(experiment_folder))

                    # Screenshot from the mock app view
                    screenshot_path = experiment_folder / "0_initial_screenshot.png"
                    st.write(
                        "DEBUG: screenshot_path:", str(screenshot_path),
                        "exists:", screenshot_path.exists()
                    )
                    if screenshot_path.exists():
                        st.session_state["last_screenshot_path"] = str(screenshot_path)

                    # Journey GIF (recording of 127.0.0.1 UI)
                    journey_gif_path = experiment_folder / "journey.gif"
                    st.write(
                        "DEBUG: journey_gif_path:", str(journey_gif_path),
                        "exists:", journey_gif_path.exists()
                    )
                    if journey_gif_path.exists():
                        st.session_state["last_journey_gif"] = str(journey_gif_path)

                    # Read experiment_data.csv and extract selected product
                    try:
                        st.write("DEBUG: Reading latest experiment_data.csv")
                        df_exp = pd.read_csv(latest_csv_path)
                        st.write("DEBUG: experiment_data.csv columns:", list(df_exp.columns))
                        st.write("DEBUG: experiment_data.csv head:")
                        st.dataframe(df_exp.head())

                        # If there's a "selected" column, prefer that row
                        if "selected" in df_exp.columns:
                            chosen = df_exp[df_exp["selected"] == 1]
                            st.write("DEBUG: #rows with selected == 1:", len(chosen))
                            if not chosen.empty:
                                row = chosen.iloc[0]
                            else:
                                row = df_exp.iloc[0]
                        else:
                            row = df_exp.iloc[0]

                        st.session_state["product_title"] = row.get("title")
                        st.session_state["image_url"] = row.get("image_url")
                        st.session_state["url"] = row.get("url")

                        st.write("DEBUG: Chosen product title:", st.session_state["product_title"])
                        st.write("DEBUG: Chosen image_url:", st.session_state["image_url"])
                        st.write("DEBUG: Chosen url:", st.session_state["url"])

                    except Exception as e:
                        st.session_state["last_result_error"] = (
                            f"Could not read experiment_data.csv: {e}"
                        )
                        st.error(st.session_state["last_result_error"])

# ===========================
# RIGHT: RESULTS PANEL
# ===========================
with right_col:
    st.subheader("I will select your product!")

    if not st.session_state["has_run"]:
        st.caption("Click 'Run Simulation' to start.")
    else:
        if st.session_state["last_result_error"]:
            st.error(st.session_state["last_result_error"])

        # 1) Mock app screenshot – "here are some options"
        if st.session_state["last_screenshot_path"]:
            st.markdown("**Here are some options for you.**")
            try:
                img = Image.open(st.session_state["last_screenshot_path"])
                w, h = img.size
                if w <= 0 or h <= 0:
                    st.warning(
                        f"Screenshot exists but has invalid dimensions ({w}x{h})."
                    )
                else:
                    st.image(
                        img,
                        caption="🖼️ Mock Shopping App View",
                        use_container_width=True,
                    )
            except Exception as e:
                st.warning(f"Could not load screenshot: {e}")

        # 3) Selected product details
        if st.session_state["product_title"]:
            st.subheader("🛍️ My Product Recommendation")
            st.markdown(f"**{st.session_state['product_title']}**")

            if st.session_state["image_url"]:
                try:
                    st.image(
                        st.session_state["image_url"],
                        caption="Selected product",
                        use_container_width=False,
                    )
                except Exception:
                    st.caption("Could not load selected product image.")

            if st.session_state["url"]:
                st.markdown(
                    f"[View on Amazon]({st.session_state['url']})"
                )

        elif not st.session_state["last_result_error"]:
            st.caption("The selected product will appear here after a successful run.")
