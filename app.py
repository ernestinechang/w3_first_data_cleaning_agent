"""Streamlit interface for the Data Cleaning Agent."""

import os

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from data_cleaning_agent import LightweightDataCleaningAgent

load_dotenv()

st.set_page_config(page_title="Data Cleaning Agent", page_icon="🧹", layout="wide")

# Amazon palette: squid ink #232F3E / #131A22, orange #FF9900 / #FEBD69,
# teal #007185, page grey #EAEDED, text #0F1111
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; max-width: 1200px;}
    header[data-testid="stHeader"] {background: transparent;}

    /* Top banner */
    .app-banner {
        background: linear-gradient(135deg, #131A22 0%, #232F3E 100%);
        border-bottom: 4px solid #FF9900;
        border-radius: 10px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(15,17,17,.25);
    }
    .app-banner h1 {color: #FFFFFF; margin: 0; padding: 0; font-size: 2rem; font-weight: 700;}
    .app-banner h1 span {color: #FF9900;}
    .app-banner p {color: #D5DBDB; margin: .35rem 0 0; font-size: 1rem;}

    /* Section headings */
    h2 {
        color: #232F3E !important;
        border-left: 5px solid #FF9900;
        padding-left: .6rem !important;
        margin-top: 1.5rem !important;
        font-size: 1.4rem !important;
    }
    h3 {color: #232F3E !important;}

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #D5D9D9;
        border-top: 3px solid #FF9900;
        border-radius: 8px;
        padding: .9rem 1.1rem;
        box-shadow: 0 1px 3px rgba(15,17,17,.12);
    }
    div[data-testid="stMetricLabel"] p {color: #565959; font-weight: 600;}
    div[data-testid="stMetricValue"] {color: #232F3E;}

    /* Buttons: Amazon yellow-orange primary, dark secondary */
    div.stButton > button[kind="primary"],
    div.stDownloadButton > button[kind="primary"] {
        background: linear-gradient(#FFD814, #F7CA00);
        border: 1px solid #FCD200;
        color: #0F1111;
        font-weight: 600;
        border-radius: 8px;
        padding: .5rem 1.6rem;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(#F7CA00, #F0B800);
        border-color: #F2C200;
        color: #0F1111;
    }
    div.stDownloadButton > button, div.stButton > button[kind="secondary"] {
        background: #232F3E;
        color: #FFFFFF;
        border: 1px solid #232F3E;
        border-radius: 8px;
        font-weight: 600;
    }
    div.stDownloadButton > button:hover {
        background: #37475A; border-color: #FF9900; color: #FFFFFF;
    }

    /* Containers, inputs, expanders, tables */
    div[data-testid="stFileUploader"] section {
        background: #FFFFFF;
        border: 2px dashed #FF9900;
        border-radius: 10px;
    }
    div[data-testid="stFileUploader"] section button {
        background: #232F3E; color: #FFFFFF; border-radius: 6px;
    }
    textarea {border-radius: 8px !important;}
    div[data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #D5D9D9;
        border-radius: 8px;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #D5D9D9;
        border-radius: 8px;
        overflow: hidden;
        background: #FFFFFF;
    }
    div[data-testid="stAlert"] {border-radius: 8px;}
    div[data-testid="stSlider"] [role="slider"] {background-color: #FF9900;}

    /* Footer */
    footer, #MainMenu {visibility: hidden;}
    </style>
    <div class="app-banner">
        <h1>🧹 Data Cleaning <span>Agent</span></h1>
        <p>Upload a CSV, let the agent plan and clean it, then review every decision.</p>
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


def show_dataset_summary(df: pd.DataFrame):
    """Headline metrics plus a per-column table."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing values", int(df.isna().sum().sum()))
    c4.metric("Duplicate rows", int(df.duplicated().sum()))
    st.dataframe(summarize_columns(df), use_container_width=True)


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

# ------------------------------------------------- 1. Dataset summary
st.header("1. Dataset summary")
show_dataset_summary(df_raw)
with st.expander("Preview raw data"):
    st.dataframe(df_raw.head(10), use_container_width=True)

# ---------------------------------------------------------- Run agent
st.header("Clean your data")
with st.container(border=True):
    instructions = st.text_area(
        "Custom instructions (optional)",
        placeholder="e.g. Keep the 'notes' column. Ages must be between 0 and 120.",
    )
    max_retries = st.slider("Max retries if the generated code fails", 0, 5, 3)
    run_clicked = st.button("Clean Data", type="primary")

if run_clicked:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set. Add it to your .env file and restart the app.")
        st.stop()

    try:
        with st.spinner("Planning and cleaning..."):
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            agent = LightweightDataCleaningAgent(model=llm, log=True)
            agent.invoke_agent(
                data_raw=df_raw,
                user_instructions=instructions.strip() or None,
                max_retries=max_retries,
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

# --------------------------------- 2 & 3. Cleaning steps and decisions
st.header("2. Cleaning steps and decisions")
if result["plan"]:
    plan_df = pd.DataFrame(result["plan"])
    plan_df["apply"] = plan_df["apply"].map({True: "✅ Applied", False: "⏭️ Skipped"})
    plan_df = plan_df.rename(columns={"step": "Step", "apply": "Decision", "reason": "Why"})
    st.dataframe(plan_df, use_container_width=True, hide_index=True)
    st.caption(
        "The decisions and reasons come from the agent's plan. Compare them with "
        "the measured before/after numbers below."
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

st.success("Done!")

# ------------------------------------------------- 4. Before and after
st.header("3. Before and after")
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
with st.expander("4. View the exact cleaning code"):
    st.code(result["code"], language="python")
    st.download_button(
        "Download code",
        data=result["code"],
        file_name="data_cleaner.py",
        mime="text/x-python",
    )
