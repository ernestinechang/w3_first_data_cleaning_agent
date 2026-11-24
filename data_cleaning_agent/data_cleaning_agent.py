# Libraries
from typing import TypedDict, List
import os
import logging

import pandas as pd
from pydantic import BaseModel, Field

from langchain.prompts import PromptTemplate
from langgraph.types import Checkpointer
from langgraph.graph import StateGraph, END

from .utils import (
    PythonOutputParser,
    get_dataframe_summary,
    execute_agent_code,
    fix_agent_code,
)

# Setup
logger = logging.getLogger(__name__)
AGENT_NAME = "lightweight_data_cleaning_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")

# Single source of truth for the cleaning steps. Used by the planner, the
# code-generation prompt and (via the plan) the Streamlit UI.
# Dataset-level steps act on the table as a whole.
DATASET_STEPS = [
    "Remove columns with more than 40% missing values",
    "Remove duplicate rows",
    "Rename columns to be more descriptive and consistent",
]
# Column-level steps are suggested and approved one column at a time.
COLUMN_STEPS = [
    "Impute missing values (mean for numeric, mode for categorical)",
    "Round to the most appropriate decimal places",
    "Correct the data type",
    "Correct obvious data entry errors (e.g., negative ages, impossible dates)",
    "Handle outliers (cap using the IQR rule)",
    "Clean text (trim whitespace, normalize case, merge spelling variants)",
]
CLEANING_STEPS = DATASET_STEPS + COLUMN_STEPS


class StepDecision(BaseModel):
    """The planner's decision for one cleaning step."""
    step: str = Field(description="The cleaning step, copied exactly from the list")
    apply: bool = Field(description="True if this step should be performed")
    reason: str = Field(description="One or two sentences citing evidence from the dataset summary")


class DatasetPlan(BaseModel):
    """Decisions for every dataset-level step."""
    decisions: List[StepDecision]


class ColumnDecisions(BaseModel):
    """Decisions for every column-level step on one column."""
    column: str = Field(description="The column name, copied exactly from the list")
    decisions: List[StepDecision]


class ColumnPlan(BaseModel):
    """Decisions for every column-level step on each requested column."""
    columns: List[ColumnDecisions]


_PLAN_PROMPT = """
You are a Data Cleaning Agent. {task}

Base every reason on evidence in the dataset summary. If the summary has no
evidence either way, say so instead of guessing. Copy step and column names
exactly. If the user instructions require or forbid a step, follow them.

Steps:
{steps}
{columns_block}
User Instructions:
{user_instructions}

Dataset Summary:
{dataset_summary}
"""


def _run_planner(model, schema, df, user_instructions, task, steps, columns=None):
    prompt = PromptTemplate(
        template=_PLAN_PROMPT,
        input_variables=["task", "steps", "columns_block", "user_instructions", "dataset_summary"],
    )
    columns_block = (
        "\nColumns:\n" + "\n".join(f"- {c}" for c in columns) + "\n" if columns else ""
    )
    return (prompt | model.with_structured_output(schema)).invoke({
        "task": task,
        "steps": "\n".join(f"- {s}" for s in steps),
        "columns_block": columns_block,
        "user_instructions": user_instructions or "None",
        "dataset_summary": get_dataframe_summary(df),
    })


def suggest_cleaning_plan(
    model, df: pd.DataFrame, user_instructions: str = None, scope: str = "all", columns=None
) -> list:
    """
    Ask the LLM which cleaning steps are needed, and why. Nothing is changed in the data.

    scope: "dataset" (whole-dataset steps only), "columns" (column-level steps for
    `columns`, default every column), or "all" (both).

    Returns a flat list of {"column": str | None, "step": str, "apply": bool,
    "reason": str} dicts. "column" is None for dataset-level steps. Every step is
    present for every requested column (anything the LLM skipped is marked not
    applied), so the UI can show a complete, editable checklist.
    """
    def row(column, step, d):
        return {
            "column": column,
            "step": step,
            "apply": d.apply if d else False,
            "reason": d.reason if d else "Not assessed by the agent.",
        }

    plan = []
    if scope in ("dataset", "all"):
        result = _run_planner(
            model, DatasetPlan, df, user_instructions,
            "For EACH dataset-level step below, say whether to apply it to this dataset and why.",
            DATASET_STEPS,
        )
        found = {d.step: d for d in result.decisions}
        plan += [row(None, s, found.get(s)) for s in DATASET_STEPS]

    if scope in ("columns", "all"):
        cols = [str(c) for c in (columns if columns is not None else df.columns)]
        result = _run_planner(
            model, ColumnPlan, df, user_instructions,
            "For EACH column listed, and for EACH column-level step below, say whether to "
            "apply the step to that column and why. Only apply a step to a column where the "
            "summary shows evidence it is needed (e.g. do not impute a column with no missing "
            "values; do not clean text on numeric columns).",
            COLUMN_STEPS, columns=cols,
        )
        found = {c.column: {d.step: d for d in c.decisions} for c in result.columns}
        for col in cols:
            plan += [row(col, s, found.get(col, {}).get(s)) for s in COLUMN_STEPS]
    return plan


class LightweightDataCleaningAgent:
    """
    LLM-powered agent that generates and executes Python code to clean pandas DataFrames.
    
    Uses an LLM to create data cleaning functions based on user instructions. The agent
    automatically retries with error correction if the generated code fails.
    
    Parameters
    ----------
    model : LLM
        Language model for generating cleaning code (e.g., ChatOpenAI).
    log : bool, default=False
        Whether to save generated code to a file.
    log_path : str, optional
        Directory for log files. Defaults to './logs/' if log=True and not specified.
    file_name : str, default="data_cleaner.py"
        Name of the log file when log=True.
    function_name : str, default="data_cleaner"
        Name of the generated cleaning function.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving agent state.
    
    Attributes
    ----------
    response : dict or None
        Stores the full response after invoke_agent() is called.
    """
    
    def __init__(
        self, 
        model, 
        log=False, 
        log_path=None, 
        file_name="data_cleaner.py", 
        function_name="data_cleaner",
        checkpointer: Checkpointer = None
    ):
        self.model = model
        self.log = log
        self.log_path = log_path
        self.file_name = file_name
        self.function_name = function_name
        self.checkpointer = checkpointer
        self.response = None
        # Build the LangGraph workflow with code generation, execution, and error fixing nodes
        self._compiled_graph = make_lightweight_data_cleaning_agent(
            model=model,
            log=log,
            log_path=log_path,
            file_name=file_name,
            function_name=function_name,
            checkpointer=checkpointer
        )
    
    def suggest_plan(self, data_raw: pd.DataFrame, user_instructions: str = None,
                     scope: str = "all", columns=None) -> list:
        """
        Suggest cleaning steps without touching the data.

        scope is "dataset", "columns" or "all"; `columns` limits a "columns" run
        to specific column names.

        Returns a list of {"step", "apply", "reason"} dicts. Review or edit it,
        then pass it to invoke_agent(cleaning_plan=...) to run exactly those steps.
        """
        return suggest_cleaning_plan(self.model, data_raw, user_instructions, scope, columns)

    def invoke_agent(self, data_raw: pd.DataFrame, user_instructions: str=None, max_retries:int=3, retry_count:int=0, cleaning_plan: list=None, **kwargs):
        """
        Generate and execute data cleaning code on the provided DataFrame.
        
        Parameters
        ----------
        data_raw : pd.DataFrame
            Raw dataset to clean.
        user_instructions : str, optional
            Custom cleaning instructions. If None, applies default cleaning steps:
            removing columns with >40% missing values, imputing missing values,
            and removing duplicates.
        max_retries : int, default=3
            Maximum number of retry attempts if generated code fails.
        retry_count : int, default=0
            Starting retry count (typically left at 0).
        **kwargs
            Additional arguments passed to the underlying graph invoke method.
        
        Returns
        -------
        None
            Results are stored in self.response and accessed via getter methods.
        """
        state = {
            "user_instructions": user_instructions,
            "data_raw": data_raw.to_dict(),
            "max_retries": max_retries,
            "retry_count": retry_count,
        }
        if cleaning_plan:
            # User-approved plan: the agent runs exactly these steps, no re-planning
            state["cleaning_plan"] = cleaning_plan
        response = self._compiled_graph.invoke(state, **kwargs)
        self.response = response  # Store full workflow response for getter methods
        return None
    
    def get_data_cleaned(self):
        """
        Retrieves the cleaned data stored after running invoke_agent.
        """
        if self.response:
            return pd.DataFrame(self.response.get("data_cleaned"))
        
    def get_data_raw(self):
        """
        Retrieves the raw data.
        """
        if self.response:
            return pd.DataFrame(self.response.get("data_raw"))
    
    def get_data_cleaner_function(self):
        """
        Retrieves the agent's cleaning function code.
        """
        if self.response:
            return self.response.get("data_cleaner_function")

    def get_cleaning_plan(self):
        """
        Retrieves the planner's decisions as a list of
        {"step": str, "apply": bool, "reason": str} dicts.
        """
        if self.response:
            return self.response.get("cleaning_plan")

    def get_error(self):
        """
        Retrieves the last execution error, or None if cleaning succeeded.
        """
        if self.response:
            return self.response.get("data_cleaner_error")


# Agent Factory Function

def make_lightweight_data_cleaning_agent(
    model, 
    log=False, 
    log_path=None, 
    file_name="data_cleaner.py",
    function_name="data_cleaner",
    checkpointer: Checkpointer = None
):
    """
    Factory function that creates a compiled LangGraph workflow for data cleaning.
    
    Builds a state graph with three nodes: code generation, execution, and error fixing.
    The workflow automatically retries with corrections if generated code fails.
    
    Parameters
    ----------
    model : LLM
        Language model for generating cleaning code.
    log : bool, default=False
        Whether to save generated code to a file.
    log_path : str, optional
        Directory for log files. Defaults to './logs/' if log=True and not specified.
    file_name : str, default="data_cleaner.py"
        Name of the log file when log=True.
    function_name : str, default="data_cleaner"
        Name of the generated cleaning function.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving workflow state.
    
    Returns
    -------
    CompiledStateGraph
        Compiled LangGraph workflow ready to process cleaning requests.
    """
    # Setup Log Directory
    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)    

    # Define state schema for the workflow graph
    class GraphState(TypedDict):
        user_instructions: str
        data_raw: dict
        data_cleaned: dict
        data_cleaner_function: str
        data_cleaner_function_path: str
        data_cleaner_function_name: str
        data_cleaner_error: str
        max_retries: int
        retry_count: int
        cleaning_plan: list

    def create_cleaning_plan(state: GraphState):
        """
        Suggest which cleaning steps this dataset needs, and why.
        Only runs when the caller did not supply an approved plan.
        """
        logger.info("Creating cleaning plan")
        df = pd.DataFrame.from_dict(state.get("data_raw"))
        return {"cleaning_plan": suggest_cleaning_plan(model, df, state.get("user_instructions"))}

    def route_start(state: GraphState):
        # A user-approved plan skips the planner entirely
        return "use_plan" if state.get("cleaning_plan") else "make_plan"

    def create_data_cleaner_code(state: GraphState):
        """
        Generate the data cleaning code based on user instructions.
        """
        logger.info("Creating data cleaner code")
        
        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)

        dataset_summary = get_dataframe_summary(df)

        # Steps come from the planner; fall back to all steps if there is no plan
        plan = state.get("cleaning_plan") or [
            {"step": s, "apply": True, "reason": ""} for s in CLEANING_STEPS
        ]
        def label(d):
            col = d.get("column")
            return f"[Column '{col}'] {d['step']}" if col else f"[Whole dataset] {d['step']}"

        steps_to_apply = "\n".join(
            f"{i}. {label(d)}" for i, d in enumerate((d for d in plan if d["apply"]), 1)
        ) or "None"
        steps_to_skip = "\n".join(
            f"- {label(d)}" for d in plan if not d["apply"] and not d.get("column")
        ) or "None"

        data_cleaning_prompt = PromptTemplate(
            template="""
            You are a Data Cleaning Agent. Create a {function_name}() function to clean the data.

            Cleaning steps to implement (in this order):
            {steps_to_apply}

            Perform ONLY the steps listed above. A step marked [Column 'x'] must
            touch only that column. Refer to columns by their ORIGINAL names, and do
            any column renaming last. Do NOT perform these dataset-level steps:
            {steps_to_skip}

            User Instructions:
            {user_instructions}

            Dataset Summary:
            {all_datasets_summary}

            Return Python code in ```python``` format with a single function:

            def {function_name}(data_raw):
                import pandas as pd
                import numpy as np
                # Your cleaning code here
                return data_cleaned

            Important: Ensure fit_transform() outputs are flattened with .ravel() when assigning to DataFrame columns.
            """,
            input_variables=[
                "user_instructions", "all_datasets_summary", "function_name",
                "steps_to_apply", "steps_to_skip",
            ]
        )

        data_cleaning_agent = data_cleaning_prompt | model | PythonOutputParser()
        
        response = data_cleaning_agent.invoke({
            "user_instructions": state.get("user_instructions") or "Follow the basic cleaning steps.",
            "all_datasets_summary": dataset_summary,
            "function_name": function_name,
            "steps_to_apply": steps_to_apply,
            "steps_to_skip": steps_to_skip,
        })
        
        # Simple logging if enabled
        file_path = None
        if log:
            file_path = os.path.join(log_path, file_name)
            with open(file_path, 'w') as f:
                f.write(response)
            logger.info(f"Code saved to: {file_path}")
   
        return {
            "data_cleaner_function": response,
            "data_cleaner_function_path": file_path,
            "data_cleaner_function_name": function_name,
        }
        
    def execute_data_cleaner_code(state):
        """
        Execute the generated cleaning code on the data.
        """
        return execute_agent_code(
            state=state,
            data_key="data_raw",
            result_key="data_cleaned",
            error_key="data_cleaner_error",
            code_snippet_key="data_cleaner_function",
            agent_function_name=state.get("data_cleaner_function_name")
        )
        
    def fix_data_cleaner_code(state: GraphState):
        """
        Fix errors in the generated data cleaning code.
        """
        data_cleaner_prompt = """
        You are a Data Cleaning Agent. Fix the broken {function_name}() function.
        
        Return Python code in ```python``` format with the corrected function definition.
        
        Broken code: 
        {code_snippet}

        Error:
        {error}
        """

        return fix_agent_code(
            state=state,
            code_snippet_key="data_cleaner_function",
            error_key="data_cleaner_error",
            llm=model,  
            prompt_template=data_cleaner_prompt,
            function_name=state.get("data_cleaner_function_name"),
        )

    # Build the workflow graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("create_cleaning_plan", create_cleaning_plan)
    workflow.add_node("create_data_cleaner_code", create_data_cleaner_code)
    workflow.add_node("execute_data_cleaner_code", execute_data_cleaner_code)
    workflow.add_node("fix_data_cleaner_code", fix_data_cleaner_code)
    
    # Set entry point
    workflow.set_conditional_entry_point(
        route_start,
        {"make_plan": "create_cleaning_plan", "use_plan": "create_data_cleaner_code"},
    )

    # Add edges
    workflow.add_edge("create_cleaning_plan", "create_data_cleaner_code")
    workflow.add_edge("create_data_cleaner_code", "execute_data_cleaner_code")
    workflow.add_edge("fix_data_cleaner_code", "execute_data_cleaner_code")
    
    # Conditional routing: retry with fixes if error occurs and retries remain
    def should_retry(state):
        has_error = state.get("data_cleaner_error") is not None
        can_retry = (
            state.get("retry_count") is not None
            and state.get("max_retries") is not None
            and state["retry_count"] < state["max_retries"]
        )
        return "fix_code" if (has_error and can_retry) else "end"
    
    workflow.add_conditional_edges(
        "execute_data_cleaner_code",
        should_retry,
        {
            "fix_code": "fix_data_cleaner_code",
            "end": END,
        },
    )
    
    # Compile the workflow
    app = workflow.compile(checkpointer=checkpointer, name=AGENT_NAME)
    
    return app
