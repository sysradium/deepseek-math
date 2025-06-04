import ast
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
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

console = Console()
error_console = Console(stderr=True, style="bold red")


class ActionType(Enum):
    THINK = "think"
    CODE = "code"
    EXECUTE = "execute"
    REFLECT = "reflect"
    ANSWER = "answer"


@dataclass
class AgentStep:
    """Represents a single step in the agent's reasoning"""

    action: ActionType
    content: str
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SolutionCandidate:
    """A candidate solution from sampling"""

    code: str
    step: AgentStep
    final_answer: Optional[Any] = None
    score: float = 0.0
    success: bool = False


@dataclass
class AgentState:
    """Current state of the agent"""

    problem: str
    history: List[AgentStep]
    current_step: int
    max_steps: int
    namespace: Dict[str, Any]
    solved: bool = False
    final_answer: Optional[Any] = None
    past_key_values: Optional[Any] = None


class MathCodeAgent:
    def _detect_device(self, requested_device: str) -> str:
        device: str = "cuda"

        if requested_device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                error_console.print(
                    Panel(
                        "[bold red]❌ ERROR: No GPU acceleration available![/bold red]\n\n"
                        "This agent requires either CUDA or MPS (Apple Silicon) for reasonable performance.\n"
                        "Available options:\n"
                        "• Install CUDA-enabled PyTorch for NVIDIA GPUs\n"
                        "• Use Apple Silicon Mac for MPS acceleration\n"
                        "• Use a cloud service with GPU support",
                        title="[bold red]🚫 Hardware Requirements Not Met[/bold red]",
                        border_style="red",
                    )
                )
                raise RuntimeError(
                    "GPU acceleration (CUDA or MPS) is required for this model"
                )

            return device

        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        elif requested_device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")

        return requested_device

    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        device: str = "auto",
        max_steps: int = 10,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        num_samples: int = 3,
    ):
        """
        Initialize the Math Code Agent with agentic loop

        Args:
            model_name: Hugging Face model to use
            device: Device to run the model on ("auto", "cuda", "mps")
            max_steps: Maximum steps in the agent loop
            temperature: Temperature for generation
            max_new_tokens: Maximum tokens to generate
            num_samples: Number of solutions to sample at each step
        """
        self.device = self._detect_device(device)
        self.max_steps = max_steps
        self.num_samples = num_samples

        with console.status(
            "[bold blue]Loading model and tokenizer...", spinner="dots"
        ):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )

            if device != "cuda":  # For MPS, manually move to device
                self.model = self.model.to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.generation_config = GenerationConfig(
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.95,
            num_return_sequences=num_samples,  # Generate multiple sequences
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        self.safe_globals = {
            "np": np,
            "numpy": np,
            "sp": sp,
            "sympy": sp,
            "Symbol": sp.Symbol,
            "symbols": sp.symbols,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "exp": sp.exp,
            "log": sp.log,
            "ln": sp.ln,
            "sqrt": sp.sqrt,
            "pi": sp.pi,
            "e": sp.E,
            "I": sp.I,
            "Matrix": sp.Matrix,
            "solve": sp.solve,
            "diff": sp.diff,
            "integrate": sp.integrate,
            "simplify": sp.simplify,
            "expand": sp.expand,
            "factor": sp.factor,
            "limit": sp.limit,
            "series": sp.series,
            "Eq": sp.Eq,
            "Rational": sp.Rational,
            "__builtins__": {
                "__import__": __import__,
                "len": len,
                "range": range,
                "str": str,
                "int": int,
                "float": float,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "pow": pow,
                "print": print,
                "bool": bool,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sorted": sorted,
                "reversed": reversed,
                "all": all,
                "any": any,
                "isinstance": isinstance,
                "type": type,
            },
        }

        self.prompt_tpl = """You are a mathematical problem solver. You must solve problems by first thinking through your approach, then providing exactly ONE self-contained Python code block at the very end.

Problem: {problem}

CRITICAL INSTRUCTIONS:
1. First, think through the problem step by step - analyze what's being asked, consider different approaches, and plan your solution
2. You may discuss multiple ideas, test different strategies mentally, but DO NOT write any code yet
3. At the very end of your response, provide exactly ONE complete, self-contained Python code block that:
   - Solves the entire problem from start to finish
   - Prints the final result as the last line (this will be your answer)
   - Uses only the available libraries: sympy (as sp), numpy (as np)
   - Is enclosed in ```python and ``` tags

RESPONSE FORMAT:
[Your thinking and analysis here...]

```python
# Your complete, self-contained solution here
# The last print statement should output the final answer
```

Remember: Think first, code once, print the answer at the end.
"""

        # Display device info with appropriate styling
        device_info = f"[yellow]{self.device.upper()}[/yellow]"
        if self.device == "cuda":
            device_info += f" (GPU: [cyan]{torch.cuda.get_device_name()}[/cyan])"
        elif self.device == "mps":
            device_info += " (Apple Silicon)"

        console.print(
            Panel.fit(
                f"[bold green]Math Code Agent Initialized[/bold green]\n"
                f"Model: [cyan]{model_name}[/cyan]\n"
                f"Device: {device_info}\n"
                f"Max Steps: [blue]{max_steps}[/blue]\n"
                f"Samples per Step: [magenta]{num_samples}[/magenta]",
                title="🤖 Agent Configuration",
            )
        )

    def _create_prompt(self, state: AgentState) -> str:
        """Create a prompt for the current state using tokenizer chat templates"""
        system_message = self.prompt_tpl.format(problem=state.problem)

        messages = [{"role": "system", "content": system_message}]

        if state.history:
            for i, step in enumerate(state.history):
                step_content = f"Step {i + 1} ({step.action.value}):\n"
                step_content += step.content
                if step.result is not None:
                    step_content += f"\nResult: {step.result}"
                if step.error:
                    step_content += f"\nError: {step.error}"

                messages.append({"role": "assistant", "content": step_content})

        messages.append(
            {"role": "user", "content": f"Continue with step {state.current_step}:"}
        )

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except (AttributeError, NotImplementedError, Exception) as e:
            console.print(
                f"[yellow]Warning: Chat template not available, using fallback formatting: {str(e)}[/yellow]"
            )
            prompt = system_message

            if state.history:
                prompt += "\n\nPrevious steps:\n"
                for i, step in enumerate(state.history):
                    prompt += f"\nStep {i + 1} ({step.action.value}):\n"
                    prompt += step.content
                    if step.result is not None:
                        prompt += f"\nResult: {step.result}"
                    if step.error:
                        prompt += f"\nError: {step.error}"

            prompt += f"\n\nContinue with step {state.current_step + 1}:"

        return prompt

    def _extract_code(self, text: str) -> str:
        """Extract Python code from the response"""
        # Look for code blocks first
        code_block_pattern = r"```python\s*(.*?)```"
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()

        # Look for lines that look like code
        lines = text.split("\n")
        code_lines = []
        in_code_block = False

        code_indicators = [
            "import ",
            "from ",
            "=",
            "def ",
            "class ",
            "if ",
            "for ",
            "while ",
            "print(",
        ]

        for line in lines:
            stripped = line.strip()
            if any(stripped.startswith(indicator) for indicator in code_indicators):
                in_code_block = True
                code_lines.append(line)
            elif in_code_block and (stripped == "" or line.startswith((" ", "\t"))):
                # Continue collecting lines that are either blank or indented
                # Use the original line to preserve indentation information
                code_lines.append(line)
            elif in_code_block:
                break

        return "\n".join(code_lines)

    def _classify_response(self, response: str) -> ActionType:
        """Classify the type of response"""
        response_lower = response.lower()

        if "```" in response or any(
            keyword in response for keyword in ["import", "def", "=", "print("]
        ):
            return ActionType.CODE
        elif any(
            keyword in response_lower for keyword in ["think", "plan", "approach"]
        ):
            return ActionType.THINK
        elif any(
            keyword in response_lower for keyword in ["answer", "result", "solution"]
        ):
            return ActionType.ANSWER
        else:
            return ActionType.THINK

    def _validate_code(self, code: str) -> bool:
        """Validate if the code is syntactically correct"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _execute_code(
        self, code: str, namespace: Dict[str, Any]
    ) -> Tuple[bool, Any, str]:
        """Execute code in a safe environment"""
        if not code.strip():
            return False, None, "Empty code"

        if not self._validate_code(code):
            return False, None, "Syntax error in code"

        # Combine safe globals with namespace
        execution_namespace = {**self.safe_globals, **namespace}

        try:
            output_buffer = StringIO()
            with redirect_stdout(output_buffer):
                exec(code, execution_namespace)

            output = output_buffer.getvalue()
            result = execution_namespace.get("final_answer")

            # Update the namespace with new variables
            for key, value in execution_namespace.items():
                if key not in self.safe_globals:
                    namespace[key] = value

            return True, result, output

        except Exception as e:
            return False, None, str(e)

    def _extract_answer(
        self,
        response: str,
        namespace: Dict[str, Any],
        output: str = "",
        success: bool = False,
    ) -> Any:
        """Extract the final answer from response or namespace"""
        if "final_answer" in namespace:
            return namespace["final_answer"]
        elif "result" in namespace:
            return namespace["result"]

        if success and output:
            last_line = output.strip().split("\n")[-1].strip()
            if last_line:
                return last_line

        answer_patterns = [
            r"(?:answer|result|solution)(?:\s*is\s*|\s*:\s*)([^\n\.]+)",
            r"final_answer\s*=\s*([^\n]+)",
            r"([0-9]+(?:\.[0-9]+)?)",
        ]

        for pattern in answer_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                try:
                    return eval(matches[-1].strip())
                except:
                    return matches[-1].strip()

        return None

    def _score_candidate(self, candidate: SolutionCandidate) -> float:
        """Score a candidate solution based on various factors"""
        score = 0.0

        # Prefer successful execution
        if candidate.success and candidate.final_answer is not None:
            score += 50

        # Code quality factors
        if candidate.code:
            if "import" in candidate.code:
                score += 5
            if "print" in candidate.code:
                score += 5
            if len(candidate.code.split("\n")) > 3:
                score += 5
            if "#" in candidate.code:
                score += 5

        # Content factors
        if candidate.step.content:
            score += len(candidate.step.content) / 100

        # Penalize errors
        if candidate.step.error:
            score -= 20

        return score

    def _generate_with_cache(self, prompt: str, state: AgentState) -> List[str]:
        """Generate responses using the model with optional caching"""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
            )

        # Decode responses
        responses = []
        input_length = inputs["input_ids"].shape[1]
        for output in outputs:
            response = self.tokenizer.decode(
                output[input_length:], skip_special_tokens=True
            )
            responses.append(response.strip())

        return responses

    def solve(self, problem: str) -> Tuple[Dict[str, Any], AgentState]:
        """Solve a mathematical problem using the agentic approach"""
        console.rule("[bold blue]🧮 Starting Problem Resolution[/bold blue]")

        problem_panel = Panel(
            problem,
            title="[bold cyan]📝 Problem Statement[/bold cyan]",
            border_style="cyan",
        )
        console.print(problem_panel)

        state = AgentState(
            problem=problem,
            history=[],
            current_step=0,
            max_steps=self.max_steps,
            namespace={},
        )

        while state.current_step < state.max_steps and not state.solved:
            state.current_step += 1

            # Create step header
            step_header = Text(
                f"Step {state.current_step}/{state.max_steps}", style="bold white"
            )
            console.print(Panel(step_header, border_style="blue"))

            prompt = self._create_prompt(state)

            with console.status(
                f"[bold yellow]Generating {self.num_samples} candidate responses...",
                spinner="dots",
            ):
                responses = self._generate_with_cache(prompt, state)

            candidates: list[SolutionCandidate] = []

            for i, response in enumerate(responses):
                candidate_title = f"[bold green]🔍 Candidate {i + 1}[/bold green]"

                response_panel = Panel(
                    response,
                    title=candidate_title,
                    border_style="green",
                    width=80,
                )
                console.print(response_panel)

                action_type = self._classify_response(response)
                console.print(
                    f"🔍 [bold green]Action Classification:[/bold green] [bold cyan]{action_type.value}[/bold cyan]"
                )

                candidate = SolutionCandidate(
                    code="", step=AgentStep(action=action_type, content=response)
                )

                if action_type == ActionType.CODE:
                    code = self._extract_code(response)
                    if code:
                        candidate.code = code
                        console.print(
                            Panel(
                                f"[dim]Extracted {len(code)} characters of code[/dim]",
                                border_style="dim",
                            )
                        )

                        console.print(
                            Panel(
                                f"[dim]{code}[/dim]",
                                title="[dim]Extracted Code[/dim]",
                                border_style="dim",
                                expand=False,
                            )
                        )

                        namespace_copy = state.namespace.copy()
                        success, result, output = self._execute_code(
                            code, namespace_copy
                        )

                        candidate.step.result = result
                        candidate.step.error = None if success else output
                        candidate.step.metadata = {
                            "execution_output": output,
                            "namespace_size": len(namespace_copy),
                        }

                        if success:
                            console.print(
                                Panel(
                                    f"✅ [bold green]Execution successful![/bold green]\n"
                                    f"Result: [cyan]{result}[/cyan]",
                                    border_style="green",
                                )
                            )
                            candidate.final_answer = result or self._extract_answer(
                                response, namespace_copy, output, success
                            )
                            candidate.success = True
                            state.namespace.update(namespace_copy)
                        else:
                            error_msg = (
                                output[:100] + "..." if len(output) > 100 else output
                            )
                            console.print(
                                Panel(
                                    f"❌ [bold red]Execution failed![/bold red]\n"
                                    f"Error: [red]{error_msg}[/red]",
                                    border_style="red",
                                )
                            )
                else:
                    answer = self._extract_answer(response, state.namespace, "", False)
                    candidate.final_answer = answer
                    if answer:
                        console.print(
                            Panel(
                                f"📝 [bold yellow]Answer extracted:[/bold yellow] [cyan]{answer}[/cyan]",
                                border_style="yellow",
                            )
                        )

                candidate.score = self._score_candidate(candidate)
                candidates.append(candidate)

            good_candidates = [c for c in candidates if c.score > 30]
            if len(good_candidates) == 0:
                console.print(
                    Panel(
                        "[bold yellow]Best answer is of very low quality, ignoring![/bold yellow]",
                        border_style="yellow",
                    )
                )
                continue

            good_candidates.sort(key=lambda x: x.score, reverse=True)
            best_candidate = good_candidates[0]

            # Display best candidate selection
            score_table = Table(title="🏆 Candidate Scoring")
            score_table.add_column("Rank", style="cyan")
            score_table.add_column("Score", style="magenta")
            score_table.add_column("Success", style="green")
            score_table.add_column("Answer", style="yellow")

            for i, candidate in enumerate(candidates):
                success_icon = "✅" if candidate.success else "❌"
                score_table.add_row(
                    f"{i + 1}",
                    f"{candidate.score:.1f}",
                    success_icon,
                    str(candidate.final_answer) if candidate.final_answer else "None",
                )

            console.print(score_table)

            state.history.append(best_candidate.step)

            successful_answers = [
                c.final_answer
                for c in candidates
                if c.success and c.final_answer is not None
            ]

            if successful_answers:
                state.solved = True
                state.final_answer = best_candidate.final_answer
                console.print(
                    Panel(
                        f"🎉 [bold green]Solution found![/bold green]\n"
                        f"Answer: [bold cyan]{state.final_answer}[/bold cyan]",
                        title="[bold green]✨ Success![/bold green]",
                        border_style="green",
                    )
                )
                break

            if len(successful_answers) > 1:
                if len({str(a) for a in successful_answers}) == 1:
                    console.print(
                        Panel(
                            "🤝 [bold green]Consensus reached among candidates![/bold green]",
                            border_style="green",
                        )
                    )
                    state.solved = True
                    state.final_answer = successful_answers[0]
                    break

        if not state.solved:
            console.print(
                Panel(
                    f"⏰ [bold yellow]Maximum steps reached ({state.max_steps})[/bold yellow]\n"
                    "No definitive solution found.",
                    title="[yellow]🔄 Process Complete[/yellow]",
                    border_style="yellow",
                )
            )

        return {
            "problem": problem,
            "solved": state.solved,
            "final_answer": state.final_answer,
            "steps": state.current_step,
            "history": state.history,
        }, state
