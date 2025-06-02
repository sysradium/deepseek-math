import ast
import logging
import re
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from enum import Enum
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sympy as sp
import torch
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()
error_console = Console(stderr=True, style="bold red")

FORMAT = "%(message)s"
def print_report(result, state):
    """Print a comprehensive report of the solving process"""
    console.rule("[bold blue]📊 SOLUTION REPORT[/bold blue]")

    # Summary table
    summary_table = Table(title="📋 Solution Summary", show_header=False)
    summary_table.add_column("Property", style="bold cyan")
    summary_table.add_column("Value", style="yellow")

    summary_table.add_row("Problem", result["problem"][:80] + "..." if len(result["problem"]) > 80 else result["problem"])
    summary_table.add_row("Status", "✅ Solved" if state.solved else "❌ Unsolved")
    summary_table.add_row("Final Answer", str(state.final_answer) if state.final_answer else "None")
    summary_table.add_row("Steps Taken", str(state.current_step))

    console.print(summary_table)

    if result["history"]:
        console.rule("[bold green]📚 Detailed Step History[/bold green]")

        for i, step in enumerate(result["history"]):
            step_content = step.content[:200] + "..." if len(step.content) > 200 else step.content

            step_style = {
                ActionType.THINK: "blue",
                ActionType.CODE: "green",
                ActionType.EXECUTE: "yellow",
                ActionType.ANSWER: "cyan"
            }.get(step.action, "white")

            step_info = f"[bold {step_style}]Step {i + 1}: {step.action.value.upper()}[/bold {step_style}]\n\n"
            step_info += step_content

            if step.result is not None:
                step_info += f"\n\n[bold green]Result:[/bold green] {step.result}"

            if step.error:
                step_info += f"\n\n[bold red]Error:[/bold red] {step.error}"

            console.print(Panel(
                step_info,
                title=f"[bold {step_style}]📝 Step {i + 1}[/bold {step_style}]",
                border_style=step_style
            ))
