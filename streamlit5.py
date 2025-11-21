import streamlit as st
import subprocess
import os
import glob
from PIL import Image
import pandas as pd

# ===========================
# CONFIG
# ===========================
st.set_page_config(page_title="Agentic AI Shopping Simulator", layout="wide")


st.markdown(
    """
    <h1>🛒 Agentic AI Shopping Simulator</h1>
    <p class="small-muted">
        1. Choose a product category, model, and prompt on the left. <br/>
        2. Click <strong>Run Simulation</strong>. <br/>
        3. The agent browses the mock shop and picks a product. The final page and selected item appear on the right.
    </p>
    """,
    unsafe_allow_html=True,
)


DEFAULT_PROMPT = (
    "You are a personal shopping assistant helping someone find a good product. "
    "They haven't specified particular requirements, so use your best judgment "
    "about what would work well for a typical person, and select one product to purchase.\n\n"
)

DATASET_ROOT = "datasets"           # base folder
LOCAL_DATA_FLAG = "--local-dataset" # correct CLI flag
MAX_EXPERIMENTS_PER_COMBO = 500     # per (category, model)

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
        data_path = os.path.join(DATASET_ROOT, f"{folder_base}.csv")

        if not os.path.exists(data_path):
            st.session_state["last_result_error"] = (
                f"Local dataset path does not exist on disk: {data_path}"
            )
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

            # -------- Run ACES with spinner (Simple runtime, local dataset) --------
            with right_col:
                with st.spinner("🤖 I am still shopping ..."):
                    # Mirrors:
                    # uv run run.py --runtime-type simple --local-dataset datasets/<category>.csv --include <model> --experiment-count-limit <n>
                    cmd = (
                        f"uv run run.py "
                        f"--runtime-type simple "
                        # f"--runtime-type screenshot "
                        f"{LOCAL_DATA_FLAG} \"{data_path}\" "
                        f"--include {model} "
                        f"--experiment-count-limit {new_count}"
                    )
                    subprocess.run(cmd, shell=True)

                if reset_happened:
                    st.info(
                        f"Reached {MAX_EXPERIMENTS_PER_COMBO} experiments for "
                        f"{category_label} / {model}. Counter reset; "
                        f"current run uses experiment-count-limit = {new_count}."
                    )

            # -------- Locate experiment_logs/<category>/.../master_experiment_* --------
            base_dir = "experiment_logs"
            run_dir = os.path.join(base_dir, category)

            if not os.path.exists(base_dir):
                st.session_state["last_result_error"] = "No experiment_logs directory found."
            elif not os.path.isdir(run_dir):
                st.session_state["last_result_error"] = (
                    f"No run directory found for category '{category}' under experiment_logs."
                )
            else:
                # Find latest experiment_data.csv under this run dir
                exp_csv_files = glob.glob(
                    os.path.join(run_dir, "**", "experiment_data.csv"),
                    recursive=True,
                )

                if not exp_csv_files:
                    st.session_state["last_result_error"] = (
                        "No experiment_data.csv files found for this category."
                    )
                else:
                    exp_csv_files_sorted = sorted(exp_csv_files, key=os.path.getmtime)
                    latest_csv = exp_csv_files_sorted[-1]
                    experiment_folder = os.path.dirname(latest_csv)  # .../master_experiment_0

                    # Screenshot from the mock app view
                    screenshot_path = os.path.join(
                        experiment_folder, "0_initial_screenshot.png"
                    )
                    if os.path.exists(screenshot_path):
                        st.session_state["last_screenshot_path"] = screenshot_path

                    # Journey GIF (recording of 127.0.0.1 UI)
                    journey_gif = os.path.join(experiment_folder, "journey.gif")
                    if os.path.exists(journey_gif):
                        st.session_state["last_journey_gif"] = journey_gif

                    # Read experiment_data.csv and extract selected product
                    try:
                        df_exp = pd.read_csv(latest_csv)

                        # If there's a "selected" column, prefer that row
                        if "selected" in df_exp.columns:
                            chosen = df_exp[df_exp["selected"] == 1]
                            if not chosen.empty:
                                row = chosen.iloc[0]
                            else:
                                row = df_exp.iloc[0]
                        else:
                            row = df_exp.iloc[0]

                        st.session_state["product_title"] = row.get("title")
                        st.session_state["image_url"] = row.get("image_url")
                        st.session_state["url"] = row.get("url")

                    except Exception as e:
                        st.session_state["last_result_error"] = (
                            f"Could not read experiment_data.csv: {e}"
                        )

# ===========================
# RIGHT: RESULTS PANEL
# ===========================
with right_col:
    st.subheader("I will select your product!")

    if not st.session_state["has_run"]:
        pass
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

        # # 2) Journey GIF – actual recorded 127.0.0.1 interaction
        # if st.session_state["last_journey_gif"]:
        #     st.markdown("---")
        #     st.markdown("**Agent interaction (recorded):**")
        #     try:
        #         st.image(
        #             st.session_state["last_journey_gif"],
        #             caption="Journey through the mock shopping app",
        #             use_container_width=True,
        #         )
        #     except Exception as e:
        #         st.warning(f"Could not load journey GIF: {e}")

        # st.markdown("---")

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
