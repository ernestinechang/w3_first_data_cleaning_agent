"""Streamlit interface for the Data Cleaning Agent."""

import os

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from data_cleaning_agent import LightweightDataCleaningAgent

load_dotenv()

st.set_page_config(page_title="Data Cleaning Agent", page_icon="🧹", layout="wide")

# Minimal Amazon-inspired style: white canvas, squid-ink text (#232F3E),
# orange (#FF9900) used sparingly for the primary action and small accents.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stApp {font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;}
    .stApp {background: #FFFFFF; color: #0F1111;}
    header[data-testid="stHeader"] {background: transparent;}
    footer, #MainMenu, [data-testid="stToolbar"], .stAppDeployButton {display: none !important;}
    h1, h2, h3, h4 {font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif !important;}
    .block-container {max-width: 1280px; padding-top: 2rem; padding-bottom: 4rem;}

    /* Masthead: text only */
    .masthead {margin-bottom: 1.5rem;}
    .masthead h1 {
        font-size: 1.75rem; font-weight: 600; letter-spacing: -0.02em;
        color: #232F3E; margin: 0; padding: 0;
    }
    .masthead p {color: #6B7280; margin: .4rem 0 0; font-size: .95rem;}
    .masthead .rule {width: 36px; height: 3px; background: #FF9900; margin-top: 1rem; border-radius: 2px;}

    /* Headings */
    h2 {
        font-size: 1.15rem !important; font-weight: 600 !important;
        letter-spacing: -0.01em; color: #232F3E !important;
        margin-top: 2rem !important; padding: 0 0 .6rem !important;
        border-bottom: 1px solid #E5E7EB;
    }
    h3 {font-size: .95rem !important; font-weight: 600 !important; color: #232F3E !important;}
    p, label, li {font-size: .92rem;}
    [data-testid="stCaptionContainer"] {color: #6B7280;}

    /* Metrics: flat, no card */
    div[data-testid="stMetric"] {
        background: transparent; border: none; border-left: 2px solid #E5E7EB;
        padding: .1rem 0 .1rem .9rem; box-shadow: none;
    }
    div[data-testid="stMetricLabel"] p {
        color: #6B7280; font-weight: 500; font-size: .75rem;
        text-transform: uppercase; letter-spacing: .05em;
    }
    div[data-testid="stMetricValue"] {color: #232F3E; font-size: 1.4rem; font-weight: 600;}

    /* Buttons */
    div.stButton > button, div.stDownloadButton > button {
        border-radius: 6px; font-weight: 500; font-size: .9rem;
        padding: .45rem 1.2rem; box-shadow: none;
        background: #FFFFFF; color: #232F3E; border: 1px solid #D1D5DB;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        border-color: #232F3E; color: #232F3E; background: #F9FAFB;
    }
    div.stButton > button[kind="primary"], div.stDownloadButton > button[kind="primary"] {
        background: #FF9900; border: 1px solid #FF9900; color: #131A22; font-weight: 600;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stDownloadButton > button[kind="primary"]:hover {
        background: #E88B00; border-color: #E88B00; color: #131A22;
    }
    div.stButton > button:disabled {opacity: .45;}

    /* Containers, inputs, tables: hairline borders, no shadows */
    div[data-testid="stVerticalBlockBorderWrapper"] {border-color: #E5E7EB !important; border-radius: 8px;}
    div[data-testid="stFileUploader"] section {
        background: #F9FAFB; border: 1px dashed #D1D5DB; border-radius: 8px;
    }
    div[data-testid="stFileUploader"] section button {background: #FFFFFF; border: 1px solid #D1D5DB;}
    textarea, input {border-radius: 6px !important;}
    div[data-testid="stExpander"] {
        background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px; box-shadow: none;
    }
    div[data-testid="stExpander"] summary {font-weight: 500; color: #232F3E;}
    div[data-testid="stDataFrame"] {border: 1px solid #E5E7EB; border-radius: 8px; overflow: hidden;}
    div[data-testid="stAlert"] {border-radius: 8px; border: 1px solid #E5E7EB; box-shadow: none;}
    div[data-testid="stAlert"][kind="info"], div[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {background: #F9FAFB !important; color: #374151 !important;}
    div[data-testid="stSlider"] [role="slider"] {background-color: #FF9900;}
    </style>
    <div class="masthead">
        <h1>Data Cleaning Agent</h1>
        <p>Upload a CSV, review the suggested cleaning steps, and decide what runs.</p>
        <div class="rule"></div>
    </div>
    """,
    unsafe_allow_html=True,
)


def summarize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column overview: type, missing values, unique values."""
    return pd.DataFrame({
        "Type": df.dtypes.astype(str),
        "Missing": df.isna().sum(),
        "Missing %": (df.isna().mean() * 100).round(1),
        "Unique": df.nunique(),
    })


# ---------------------------------------------------------------- Upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if not uploaded_file:
    st.info("Upload a CSV file to get started.")
    st.stop()

try:
    df_raw = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read this file as a CSV: {e}")
    st.stop()

if df_raw.empty:
    st.error("The uploaded file has no rows.")
    st.stop()

# A new file invalidates results from a previous run
file_key = f"{uploaded_file.name}-{uploaded_file.size}"
if st.session_state.get("file_key") != file_key:
    st.session_state["file_key"] = file_key
    st.session_state.pop("result", None)
    st.session_state.pop("dataset_plan", None)
    st.session_state["column_plans"] = {}
    st.session_state["column_edits"] = {}
    st.session_state["view_col"] = None

# ------------------------------------------------------------ helpers
STEP_COLUMNS = ["Apply", "Step", "Why suggested"]
st.session_state.setdefault("ver", {})


def bump(key: str):
    """New suggestions replace the user's edits, so give the editor a fresh state."""
    st.session_state["ver"][key] = st.session_state["ver"].get(key, 0) + 1



def step_table(plan_rows: list, key: str, step_label: str = "Step (editable)") -> list:
    """Show an editable checklist of steps and return the user's version as dicts."""
    table = pd.DataFrame(
        [{"Apply": r["apply"], "Step": r["step"], "Why suggested": r["reason"]} for r in plan_rows],
        columns=STEP_COLUMNS,
    )
    edited = st.data_editor(
        table,
        key=f"{key}_{st.session_state['ver'].get(key, 0)}",
        num_rows="dynamic",  # lets the user add or delete steps
        use_container_width=True,
        hide_index=True,
        column_config={
            "Apply": st.column_config.CheckboxColumn("Apply", default=True, width="small"),
            "Step": st.column_config.TextColumn(step_label, required=True, width="large"),
            "Why suggested": st.column_config.TextColumn("Why suggested", width="large"),
        },
    )
    return [
        {
            "step": str(r["Step"]).strip(),
            "apply": bool(r["Apply"]) if pd.notna(r["Apply"]) else True,
            "reason": r["Why suggested"] if pd.notna(r["Why suggested"]) else "Added by user",
        }
        for _, r in edited.iterrows()
        if pd.notna(r["Step"]) and str(r["Step"]).strip()
    ]


def show_column_profile(df: pd.DataFrame, col: str):
    """Compact profile of one column: type, missing, unique, examples, numeric range."""
    s = df[col]
    r1a, r1b = st.columns(2)
    r2a, r2b = st.columns(2)
    r1a.metric("Type", str(s.dtype))
    r1b.metric("Missing", f"{int(s.isna().sum())} ({s.isna().mean() * 100:.1f}%)")
    r2a.metric("Unique", int(s.nunique()))
    if pd.api.types.is_numeric_dtype(s) and s.notna().any():
        r2b.metric("Min / Max", f"{s.min():.4g} / {s.max():.4g}")
    else:
        r2b.metric("Most common", str(s.mode().iloc[0]) if s.notna().any() else "n/a")
    examples = s.dropna().astype(str).unique()[:6].tolist()
    st.caption("Example values: " + (", ".join(f"`{e}`" for e in examples) or "none"))


def run_suggestion(scope: str, columns=None):
    """Ask the agent for suggestions and store them. Returns True on success."""
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set. Add it to your .env file and restart the app.")
        return False
    try:
        with st.spinner("Analysing your data..."):
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            plan = LightweightDataCleaningAgent(model=llm).suggest_plan(
                df_raw, instructions.strip() or None, scope=scope, columns=columns
            )
    except Exception as e:
        st.error(f"Could not suggest steps: {e}")
        return False
    if scope == "dataset":
        st.session_state["dataset_plan"] = plan
        bump("dataset_steps")
    else:
        for col in {r["column"] for r in plan}:
            st.session_state["column_plans"][col] = [r for r in plan if r["column"] == col]
            st.session_state["column_edits"].pop(col, None)
        st.session_state["view_col"] = None  # force the open editor to reload
    st.session_state.pop("result", None)  # old results no longer match
    return True


st.session_state.setdefault("column_plans", {})
st.session_state.setdefault("dataset_plan", None)
st.session_state.setdefault("column_edits", {})

# ------------------------------- 1. Dataset summary and dataset-level steps
st.header("1. Dataset overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", df_raw.shape[0])
m2.metric("Columns", df_raw.shape[1])
m3.metric("Missing values", int(df_raw.isna().sum().sum()))
m4.metric("Duplicate rows", int(df_raw.duplicated().sum()))

left, right = st.columns([2, 3], gap="large")
with left:
    st.markdown("**Columns at a glance**")
    st.dataframe(summarize_columns(df_raw), use_container_width=True, height=260)
    with st.expander("Preview raw data"):
        st.dataframe(df_raw.head(10), use_container_width=True)
with right:
    instructions = st.text_area(
        "Custom instructions (optional)",
        placeholder="e.g. Keep the 'notes' column. Ages must be between 0 and 120.",
        help="Used when suggesting steps and when writing the cleaning code.",
        height=80,
    )
    head_l, head_r = st.columns([3, 2])
    head_l.markdown("**Whole-dataset steps**")
    if head_r.button("Suggest steps", key="suggest_dataset", use_container_width=True):
        run_suggestion("dataset")
    approved_plan = []
    if st.session_state["dataset_plan"]:
        for r in step_table(st.session_state["dataset_plan"], "dataset_steps"):
            approved_plan.append({**r, "column": None})
    else:
        st.caption("The agent suggests; you decide. Nothing changes until you approve.")

# ----------------------------------------- 2. Column by column
st.header("2. Column by column")
pick_l, pick_r = st.columns([3, 1])


col_names = [str(c) for c in df_raw.columns]
labels = {}
for c in col_names:
    rows = st.session_state["column_plans"].get(c)
    n = f"  ·  {sum(r['apply'] for r in rows)} selected" if rows else ""
    labels[c] = f"{c}  ·  {df_raw[c].dtype}{n}"
col = pick_l.selectbox("Column", col_names, format_func=labels.get)
pick_r.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
if pick_r.button("Suggest for all columns", use_container_width=True):
    run_suggestion("columns")

profile_col, steps_col = st.columns([2, 3], gap="large")
with profile_col:
    show_column_profile(df_raw, col)
    if st.button(
        "Re-suggest for this column"
        if st.session_state["column_plans"].get(col)
        else "Suggest for this column",
        key=f"suggest_{col}",
        use_container_width=True,
    ):
        run_suggestion("columns", [col])

with steps_col:
    if not st.session_state["column_plans"].get(col):
        st.caption("No steps suggested yet for this column.")
    else:
        # Only the selected column's editor is on screen, and Streamlit drops the
        # state of widgets that are not rendered. So edits are saved explicitly,
        # and the editor restarts from the saved version when the column changes.
        if st.session_state.get("view_col") != col:
            st.session_state["view_base"] = (
                st.session_state["column_edits"].get(col)
                or st.session_state["column_plans"][col]
            )
            st.session_state["view_col"] = col
            bump(f"col_{col}")
        st.session_state["column_edits"][col] = step_table(
            st.session_state["view_base"], f"col_{col}"
        )

# Every column with steps counts, not just the one being viewed
for c in col_names:
    rows = st.session_state["column_edits"].get(c) or st.session_state["column_plans"].get(c)
    for r in rows or []:
        approved_plan.append({**r, "column": c})

if not approved_plan:
    st.info("Suggest steps above, then review and edit them to enable cleaning.")
    st.stop()

# ----------------------------------------------------------- Run agent
n_apply = sum(p["apply"] for p in approved_plan)
with st.container(border=True):
    slider_col, button_col = st.columns([3, 2], vertical_alignment="bottom")
    max_retries = slider_col.slider("Max retries if the generated code fails", 0, 5, 3)
    run_clicked = button_col.button(
        f"Clean Data ({n_apply} step{'s' if n_apply != 1 else ''} selected)",
        type="primary",
        disabled=n_apply == 0,
        use_container_width=True,
    )

if run_clicked:
    try:
        with st.spinner("Cleaning..."):
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            agent = LightweightDataCleaningAgent(model=llm, log=True)
            agent.invoke_agent(
                data_raw=df_raw,
                user_instructions=instructions.strip() or None,
                max_retries=max_retries,
                cleaning_plan=approved_plan,
            )
        # Keep results in session_state so they survive Streamlit reruns
        # (e.g. clicking the download button)
        st.session_state["result"] = {
            "cleaned": agent.get_data_cleaned(),
            "plan": agent.get_cleaning_plan(),
            "code": agent.get_data_cleaner_function(),
            "error": agent.get_error(),
        }
    except Exception as e:
        st.session_state.pop("result", None)
        st.error(f"The agent failed unexpectedly: {e}")

result = st.session_state.get("result")
if not result:
    st.stop()

df_cleaned = result["cleaned"]
failed = result["error"] is not None or df_cleaned is None or df_cleaned.empty

# --------------------------------- 3. Steps that were run
st.header("3. Steps that were run")
if result["plan"]:
    plan_df = pd.DataFrame(result["plan"])
    plan_df["column"] = plan_df["column"].fillna("Whole dataset")
    plan_df = plan_df[["column", "step", "apply", "reason"]]
    plan_df["apply"] = plan_df["apply"].map({True: "Applied", False: "Skipped"})
    plan_df = plan_df.rename(columns={"column": "Scope", "step": "Step", "apply": "Decision", "reason": "Why"})
    st.dataframe(plan_df, use_container_width=True, hide_index=True)
    st.caption(
        "These are the steps you approved. Compare them with the measured "
        "before/after numbers below."
    )
else:
    st.warning("No cleaning plan was returned.")

# --------------------------------------------------- Failure handling
if failed:
    st.error("Cleaning failed after all retries.")
    if result["error"]:
        st.code(result["error"])
    if result["code"]:
        with st.expander("Last generated code"):
            st.code(result["code"], language="python")
    st.stop()

st.success("Cleaning complete.")

# ------------------------------------------------- 4. Before and after
st.header("4. Before and after")
m1, m2, m3, m4 = st.columns(4)
rows_b, rows_a = df_raw.shape[0], df_cleaned.shape[0]
cols_b, cols_a = df_raw.shape[1], df_cleaned.shape[1]
miss_b, miss_a = int(df_raw.isna().sum().sum()), int(df_cleaned.isna().sum().sum())
dup_b, dup_a = int(df_raw.duplicated().sum()), int(df_cleaned.duplicated().sum())
# delta_color="off": whether a change is good depends on the metric
m1.metric("Rows", rows_a, delta=rows_a - rows_b, delta_color="off")
m2.metric("Columns", cols_a, delta=cols_a - cols_b, delta_color="off")
m3.metric("Missing values", miss_a, delta=miss_a - miss_b, delta_color="inverse")
m4.metric("Duplicate rows", dup_a, delta=dup_a - dup_b, delta_color="inverse")

left, right = st.columns(2)
with left:
    st.subheader("Before")
    st.dataframe(df_raw.head(10), use_container_width=True)
with right:
    st.subheader("After")
    st.dataframe(df_cleaned.head(10), use_container_width=True)

renamed_note = [
    f"`{b}` → `{a}`" for b, a in zip(df_raw.columns, df_cleaned.columns)
    if b != a and cols_a == cols_b
]
if renamed_note:
    st.caption("Renamed columns: " + ", ".join(renamed_note))

st.download_button(
    "Download Cleaned Data",
    data=df_cleaned.to_csv(index=False),
    file_name="cleaned_data.csv",
    mime="text/csv",
    type="primary",
)

# ------------------------------------------------------ 5. Cleaning code
with st.expander("5. View the exact cleaning code"):
    st.code(result["code"], language="python")
    st.download_button(
        "Download code",
        data=result["code"],
        file_name="data_cleaner.py",
        mime="text/x-python",
    )
