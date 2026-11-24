# Data Cleaning Agent

An AI-powered data cleaning agent that cleans messy datasets using LangChain and LangGraph. The agent **suggests** cleaning steps for the whole dataset and for each column, **you decide** which to apply (and can edit or add your own), and only then does it generate and run Python code. A Streamlit web app walks you through every decision before you download the result.

## Features

- **You stay in control**: The agent only suggests. Every step can be ticked on or off, reworded, deleted, or supplemented with your own. Nothing touches your data until you click Clean Data.
- **Dataset-level and column-level steps**: Review whole-dataset steps first, then pick a column to see its profile (type, missing values, unique values, examples, range) next to its own editable step list. Each section has its own suggest button, and you can ask for suggestions for all columns or just one.
- **Evidence-based suggestions**: Every suggestion comes with a reason citing the data, and steps that don't apply to a column are left unticked.
- **Nine cleaning steps** (six new, see [Cleaning Steps](#cleaning-steps)).
- **Custom instructions**: Tell the agent what to keep, skip or enforce (e.g. "Keep the `notes` column. Ages must be between 0 and 120.").
- **Transparent results**: See the steps that were run with their scope, measured before/after metrics (rows, columns, missing values, duplicates), and side-by-side previews.
- **Exportable output**: Download the cleaned CSV and the exact Python cleaning code that produced it.
- **Automatic retries**: If the generated code fails, the agent fixes it and retries (0-5 attempts, configurable in the UI). Failures show the last error and the last generated code.
- **Richer dataset profiling**: The LLM sees shape, duplicate count, example values per column and numeric ranges, which helps it spot placeholders, mixed formats and spelling variants.
- **Minimal, wide UI**: A clean white layout with Amazon navy and orange accents, arranged side by side to keep scrolling to a minimum. Styling lives in `.streamlit/config.toml` plus custom CSS in `app.py`.

## How It Works

1. **Suggest**: On request, the LLM reviews a profile of your dataset and suggests, for the whole dataset and/or for each column, which steps to apply and why. Nothing is changed.
2. **Review**: You edit the suggestions in the app: tick or untick, reword, add or delete steps, at dataset level and per column.
3. **Generate**: The LLM writes a cleaning function covering only your approved steps. A column step may only touch its own column.
4. **Execute**: The function runs on your data.
5. **Retry**: If execution fails, the error goes back to the LLM to fix the code, up to the retry limit.

## Cleaning Steps

The original agent always applied three fixed steps to the whole dataset. The agent now offers nine, split by scope. You choose which ones run.

**Whole-dataset steps**

| Step | Status |
|------|--------|
| Remove columns with more than 40% missing values | Original |
| Remove duplicate rows | Original |
| Rename columns to be more descriptive and consistent | **New** |

**Column-level steps** (suggested and approved per column)

| Step | Status |
|------|--------|
| Impute missing values (mean for numeric, mode for categorical) | Original (was dataset-wide) |
| Round to the most appropriate decimal places | **New** |
| Correct the data type | **New** |
| Correct obvious data entry errors (e.g. negative ages, impossible dates) | **New** |
| Handle outliers (cap using the IQR rule) | **New** |
| Clean text (trim whitespace, normalize case, merge spelling variants) | **New** |

Only approved steps are passed to the code generator. Columns are referred to by their original names, and renaming runs last. The lists live in `DATASET_STEPS` and `COLUMN_STEPS` in `data_cleaning_agent/data_cleaning_agent.py`; edit them and the planner, code generator and UI all pick up the change.

## Setup

### Prerequisites

- **Python 3.9 or higher** (3.9, 3.10, 3.11, 3.12, or 3.13) - **Note**: Python 3.9.7 is not supported due to a Streamlit compatibility issue
- **Poetry** (dependency manager)
- **OpenAI API Key**

### Installation Steps

1. **Install Poetry** (if not already installed):
   
   **Windows (PowerShell)**:
   ```powershell
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
   ```
   
   **macOS/Linux**:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   
   After installation, restart your terminal. If `poetry` command is not found:
   - **Windows**: Add `%APPDATA%\Python\Scripts` to your system PATH
   - **macOS/Linux**: Add `export PATH="$HOME/.local/bin:$PATH"` to your `~/.bashrc` or `~/.zshrc`

2. **Install dependencies**:
   ```bash
   poetry install
   ```
   
   This will install all dependencies with the exact versions specified in `poetry.lock`, ensuring consistency across all environments.

3. **Set up your OpenAI API key**:
   
   **Windows**:
   ```powershell
   copy .env.example .env
   ```
   
   **macOS/Linux**:
   ```bash
   cp .env.example .env
   ```
   
   Then edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

### Multiple Python Versions?

If you have multiple Python versions installed and want to use a specific one:

```bash
# Tell Poetry which Python to use
poetry env use python3.11  # or python3.9, python3.10, python3.12, etc.

# Then install dependencies
poetry install
```

Poetry will create a virtual environment with your chosen Python version.

## Usage

### Streamlit Web Interface

```bash
poetry run streamlit run app.py
```

1. **Dataset overview**: Upload a CSV and review the summary. Optionally add custom instructions.
2. Click **Suggest steps** to get whole-dataset recommendations, then tick, reword, add or delete steps in the table.
3. **Column by column**: Pick a column to see its profile. Click **Suggest for this column** (or **Suggest for all columns**) and edit its steps. Your edits are kept when you switch columns.
4. Click **Clean Data** (the button shows how many steps you selected across the dataset and all columns).
5. Review the steps that were run and the before/after comparison.
6. Download the cleaned CSV and, if you want, the cleaning code.

Re-suggesting replaces your edits for that part (the dataset steps, or the column you re-suggest). A sample file is available at `data/sample_data.csv`.

### Python API

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from data_cleaning_agent import LightweightDataCleaningAgent

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = LightweightDataCleaningAgent(model=llm)
df = pd.read_csv("your_data.csv")

# 1. Get suggestions. Nothing is changed in the data.
# scope: "all" (default), "dataset", or "columns" (optionally columns=["age"])
plan = agent.suggest_plan(df, user_instructions="Keep the 'notes' column.")
# [{"column": None | "age", "step": ..., "apply": True/False, "reason": ...}, ...]

# 2. Review or edit the plan, e.g. skip outlier handling on one column
for item in plan:
    if item["column"] == "income" and "outliers" in item["step"]:
        item["apply"] = False

# 3. Run exactly the approved steps
agent.invoke_agent(data_raw=df, cleaning_plan=plan, max_retries=3)

cleaned_df = agent.get_data_cleaned()
code = agent.get_data_cleaner_function()
error = agent.get_error()               # None if cleaning succeeded
```

Calling `invoke_agent(data_raw=df)` without a `cleaning_plan` makes the agent plan and clean in one go, as before.

## Deployment

### Docker

The repo includes a `Dockerfile` (Python 3.11, dependencies installed from `poetry.lock`, runs as a non-root user).

1. **Build the image**:
   ```bash
   docker build -t data-cleaning-agent .
   ```
2. **Run it**, passing your API key at run time (it is never baked into the image):
   ```bash
   docker run --rm -p 8501:8501 -e OPENAI_API_KEY=sk-your-key-here data-cleaning-agent
   ```
   Or read it from your `.env` file:
   ```bash
   docker run --rm -p 8501:8501 --env-file .env data-cleaning-agent
   ```
3. Open http://localhost:8501.

`.dockerignore` keeps `.env`, `.git` and `logs/` out of the image. To keep the generated cleaning code between runs, mount the logs folder: add `-v "$(pwd)/logs:/app/logs"` to the `docker run` command (on Linux, make that folder writable by UID 1000 or run the container with `--user "$(id -u):$(id -g)"`).

The same image can be deployed to any container host (Render, Fly.io, AWS, etc.): set `OPENAI_API_KEY` as an environment variable and route traffic to port 8501.

### Streamlit Community Cloud (free)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), click **Create app**, and select your repo, the `main` branch and `app.py` as the main file.
3. Under **Advanced settings**, choose a supported Python version (3.11 or 3.12 recommended).
4. Add your API key under **Secrets**:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```
   Top-level secrets are exposed to the app as environment variables, so no code change is needed.
5. Click **Deploy**.

If your Poetry dependencies aren't picked up, export a `requirements.txt` and commit it:
```bash
poetry export --without-hashes -o requirements.txt
```
(On Poetry 2+, first run `poetry self add poetry-plugin-export`.)

### Self-hosting

```bash
poetry install --only main
poetry run streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Set `OPENAI_API_KEY` in the environment or a `.env` file, and put the app behind a reverse proxy with HTTPS.

### Deployment notes

- **Never commit `.env` or API keys.** `.env` is gitignored; use your platform's secrets manager.
- **Cost**: Each run makes a few OpenAI calls (plan, code generation, plus any retries). Consider usage limits on your key if the app is public.
- **Security**: The agent executes LLM-generated code with `exec()`. Only deploy publicly with authentication, or in a sandboxed container, and treat uploaded data as untrusted.

## Project Structure

```
data-cleaning-agent/
├── .streamlit/
│   └── config.toml             # UI theme (Amazon palette)
├── data/
│   └── sample_data.csv         # Sample messy dataset
├── data_cleaning_agent/
│   ├── __init__.py
│   ├── data_cleaning_agent.py  # Agent: planning, code generation, execution, retries
│   └── utils.py                # Dataset profiling and code execution helpers
├── app.py                      # Streamlit interface
├── Dockerfile                  # Container image
├── .dockerignore               # Files excluded from the image
├── pyproject.toml
├── poetry.lock                 # Committed for reproducible installs
└── README.md
```

**Important**: The `poetry.lock` file is committed to ensure all users get identical, tested dependency versions.
