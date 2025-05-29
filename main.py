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
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


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
    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        device: str = "cuda",
        max_steps: int = 10,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        num_samples: int = 3,
    ):
        """
        Initialize the Math Code Agent with agentic loop

        Args:
            model_name: Hugging Face model to use
            device: Device to run the model on
            max_steps: Maximum steps in the agent loop
            temperature: Temperature for generation
            max_new_tokens: Maximum tokens to generate
            num_samples: Number of solutions to sample at each step
        """
        self.device = device
        self.max_steps = max_steps
        self.num_samples = num_samples

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

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

        self.prompt_tpl = """You are a mathematical problem solver. Solve problems step by step using Python code.

Problem: {problem}

Instructions:
1. Think about the problem and plan your approach
2. Write Python code to solve it
3. Store the final answer in a variable called 'final_answer'
4. Use print statements to show your work

Available libraries: sympy (as sp), numpy (as np)
"""

    def _create_prompt(self, state: AgentState) -> str:
        """Create prompt based on current state"""

        prompt = self.prompt_tpl.format(problem=state.problem)

        history = []
        for i, step in enumerate(state.history):
            history.append(f"{step.action.value}: {step.content}")
            if step.result is not None:
                history.append(f"Result: {step.result}")
            if step.error:
                history.append(f"Error: {step.error}")

        if len(history):
            prompt += "Previous steps:\n" + ("\n".join(history))

        if state.namespace:
            variables = {
                k: v
                for k, v in state.namespace.items()
                if k not in self.safe_globals and not k.startswith("__")
            }
            if variables:
                prompt += f"\n\nCurrent variables: {list(variables.keys())}"

        if state.current_step == 1:
            prompt += "\n\nFirst, think about how to solve this problem:"
        elif state.history and state.history[-1].error:
            prompt += "\n\nThe previous attempt had an error. Try a different approach:"
        elif state.history and state.history[-1].action == ActionType.CODE:
            prompt += "\n\nNow execute and verify your solution:"
        else:
            prompt += "\n\nContinue solving the problem. Write Python code:"

        return prompt

    def _extract_code(self, text: str) -> Optional[str]:
        """Extract Python code from text"""
        code_pattern = r"```python\n(.*?)```"
        matches = re.findall(code_pattern, text, re.DOTALL)
        if matches:
            return matches[-1].strip()

        # Look for code patterns
        lines = text.split("\n")
        code_lines = []
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
            if any(indicator in line for indicator in code_indicators):
                code_lines.append(line)
            elif code_lines and line.startswith((" ", "\t")):  # Indented line
                code_lines.append(line)

        if code_lines:
            return "\n".join(code_lines)

        return None

    def _classify_response(self, text: str) -> ActionType:
        """Classify the type of response"""
        text_lower = text.lower()

        if "```python" in text or self._extract_code(text):
            return ActionType.CODE

        answer_indicators = [
            "final answer",
            "the answer is",
            "therefore",
            "solution:",
            "result:",
        ]
        if any(indicator in text_lower for indicator in answer_indicators):
            return ActionType.ANSWER

        if any(
            word in text_lower for word in ["check", "verify", "validate", "correct"]
        ):
            return ActionType.REFLECT

        return ActionType.THINK

    def _validate_code(self, code: str) -> Tuple[bool, str]:
        """Validate Python code before execution"""
        try:
            dangerous_patterns = [
                r"__import__",
                r"exec\s*\(",
                r"eval\s*\(",
                r"compile\s*\(",
                r"open\s*\(",
                r"file\s*\(",
                r"input\s*\(",
                r"raw_input\s*\(",
                r"os\.",
                r"sys\.",
                r"subprocess",
                r"requests",
                r"urllib",
            ]

            for pattern in dangerous_patterns:
                if re.search(pattern, code):
                    return False, f"Dangerous operation detected: {pattern}"

            ast.parse(code)
            return True, "Code is valid"

        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _execute_code(
        self, code: str, namespace: Dict[str, Any]
    ) -> Tuple[bool, Any, str]:
        """Execute Python code in the given namespace"""
        is_valid, validation_msg = self._validate_code(code)
        if not is_valid:
            return False, None, validation_msg

        exec_namespace = self.safe_globals.copy()
        exec_namespace.update(namespace)

        try:
            output_buffer = StringIO()

            with redirect_stdout(output_buffer):
                exec(code, exec_namespace)

            for key, value in exec_namespace.items():
                if key not in self.safe_globals and not key.startswith("__"):
                    namespace[key] = value

            final_answer = exec_namespace.get("final_answer", None)
            output = output_buffer.getvalue()

            return True, final_answer, output

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            return False, None, error_msg

    def _extract_answer(self, text: str, namespace: Dict[str, Any]) -> Optional[Any]:
        """Extract final answer from text or namespace"""
        if "final_answer" in namespace:
            return namespace["final_answer"]

        text_lower = text.lower()

        patterns = [
            r"final answer\s*[:=]\s*(.+?)(?:\.|$)",
            r"answer\s*[:=]\s*(.+?)(?:\.|$)",
            r"result\s*[:=]\s*(.+?)(?:\.|$)",
            r"therefore\s*[:,]\s*(.+?)(?:\.|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text_lower, re.MULTILINE)
            if match:
                answer_text = match.group(1).strip()
                try:
                    return sp.sympify(answer_text)
                except:
                    return answer_text

        return None

    def _score_candidate(self, candidate: SolutionCandidate, problem: str) -> float:
        """Score a solution candidate"""
        score = 0.0

        if candidate.final_answer is not None:
            score += 40

        code_steps = []
        if (
            candidate.step.action == ActionType.CODE
            and candidate.step.result is not None
        ):
            code_steps = [candidate.step]
        score += len(code_steps) * 10

        if candidate.code:
            if "import" in candidate.code:
                score += 5
            if "print" in candidate.code:
                score += 5
            if len(candidate.code.split("\n")) > 3:
                score += 5
            if "#" in candidate.code:
                score += 5

        if candidate.step.action == ActionType.THINK:
            score += 10

        score -= 10 if candidate.step.error else 0

        return max(0, score)

    def _generate_with_cache(self, prompt: str, state: AgentState) -> List[str]:
        """Generate multiple responses using KV cache if available"""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_kwargs = {
            "generation_config": self.generation_config,
            "return_dict_in_generate": True,
            "output_hidden_states": False,
            "output_attentions": False,
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        return [
            self.tokenizer.decode(
                outputs.sequences[i][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            for i in range(self.num_samples)
        ]

    def solve(self, problem: str) -> Dict[str, Any]:
        """
        Solve a mathematical problem using agentic loop with sampling

        Args:
            problem: The mathematical problem to solve

        Returns:
            Dictionary containing the solution and solving process
        """
        state = AgentState(
            problem=problem,
            history=[],
            current_step=0,
            max_steps=self.max_steps,
            namespace={},
        )

        while state.current_step < state.max_steps and not state.solved:
            state.current_step += 1
            print(f"\n--- Step {state.current_step}/{state.max_steps} ---")

            prompt = self._create_prompt(state)

            print(f"Generating {self.num_samples} responses...")
            responses = self._generate_with_cache(prompt, state)

            candidates = []
            for i, response in enumerate(responses):
                print(f"\nCandidate {i + 1}:")
                print(response[:200] + "..." if len(response) > 200 else response)

                action_type = self._classify_response(response)

                candidate = SolutionCandidate(
                    code="", step=AgentStep(action=action_type, content=response)
                )

                if action_type == ActionType.CODE:
                    code = self._extract_code(response)
                    if code:
                        candidate.code = code
                        print(f"Extracted code ({len(code)} chars)")

                        namespace_copy = state.namespace.copy()
                        success, result, output = self._execute_code(
                            code, namespace_copy
                        )

                        candidate.step.result = result
                        candidate.step.error = None if success else output
                        candidate.step.metadata = {
                            "code": code,
                            "output": output,
                            "namespace": namespace_copy,
                        }

                        if success:
                            print(f"Execution successful. Result: {result}")
                            candidate.final_answer = result or self._extract_answer(
                                response, namespace_copy
                            )
                        else:
                            print(f"Execution failed: {output[:100]}")

                elif action_type == ActionType.ANSWER:
                    answer = self._extract_answer(response, state.namespace)
                    candidate.final_answer = answer
                    print(f"Extracted answer: {answer}")

                candidates.append(candidate)

            for candidate in candidates:
                candidate.score = self._score_candidate(candidate, problem)
                candidate.success = candidate.final_answer is not None

            candidates.sort(key=lambda c: c.score, reverse=True)

            best_candidate = candidates[0]
            print(f"\nBest candidate score: {best_candidate.score}")

            state.history.append(best_candidate.step)

            if best_candidate.code and best_candidate.step.result is not None:
                if "namespace" in best_candidate.step.metadata:
                    state.namespace.update(best_candidate.step.metadata["namespace"])

            if best_candidate.final_answer is not None:
                state.solved = True
                state.final_answer = best_candidate.final_answer
                print(f"\nSolution found: {state.final_answer}")

            successful_answers = [
                c.final_answer for c in candidates if c.final_answer is not None
            ]
            if len(successful_answers) >= 2:
                if lean({str(a) for a in successful_answers}) == 1:
                    print("Consensus reached among candidates!")
                    state.solved = True
                    state.final_answer = successful_answers[0]

        result = {
            "problem": problem,
            "solved": state.solved,
            "final_answer": state.final_answer,
            "steps": len(state.history),
            "history": state.history,
            "namespace": {
                k: str(v)
                for k, v in state.namespace.items()
                if k not in self.safe_globals and not k.startswith("__")
            },
        }

        return result, state


mini_bench = [
    {"question": "Calculate 1+1", "answer": 2},
    {"question": "Compute 48*24", "answer": 1152},
    {
        "question": "The numbers 1, 2, 7, 9, 10, 15, and 19 are written on the board. Taking turns, Alice and Bob each erase one number until only one number remains. The sum of the numbers erased by Alice is double the sum of those erased by Bob. Which number remained written?",
        "answer": 9,
    },
    {
        "question": "Calculate 74.2989498989 / 24.29898984 ** 2",
        "answer": 0.12583638597408814,
    },
    {
        "question": "Here is a polynomial: x**3 - 6x**2 - 16x = 0. one root is -2, what is the largest of the other 2 roots?",
        "answer": 8,
    },
    {
        "question": "What is the 12th number of the Fibonacci sequence if the first is 1 and the second is 1?",
        "answer": 17711,
    },
    {"question": "Calculate -3 to the power of 12.4", "answer": -824714.363647153},
    {"question": "Divide 999999 by 7", "answer": 142857.0},
    {
        "question": "What is the square root of 123456789?",
        "answer": 11111.111060555555,
    },
    {"question": "Compute 2 to the power of 0.256", "answer": 1.1941631870745895},
]


def print_report(result, state):
    print("\n" + "=" * 50)
    print("SOLUTION SUMMARY")
    print(f"Problem: {problem}")
    print(f"Solved: {state.solved}")
    print(f"Final Answer: {state.final_answer}")
    print(f"Steps taken: {state.current_step}")

    print("\n\nDetailed History:")
    for i, step in enumerate(result["history"]):
        print(f"\nStep {i + 1} ({step.action.value}):")
        print(f"Content:\n{step.content}")
        if step.result is not None:
            print(f"Result: {step.result}")
        if step.error:
            print(f"Error: {step.error}")


if __name__ == "__main__":
    agent = MathCodeAgent(
        model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        max_steps=8,
        temperature=0.7,
        num_samples=2,
        max_new_tokens=1024,
        device="mps",
    )

    problem = mini_bench[4]["question"]

    result, agent_state = agent.solve(problem)

    print_report(result, agent_state)
