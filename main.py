from rich.console import Console
from rich.panel import Panel

from agent import MathCodeAgent
from report import print_report

console = Console()
error_console = Console(stderr=True, style="bold red")

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


if __name__ == "__main__":
    console.print(
        Panel(
            "[bold green]🚀 Math Code Agent Starting...[/bold green]",
            title="[bold blue]🤖 DeepSeek Math Agent[/bold blue]",
            border_style="blue",
        )
    )

    agent = MathCodeAgent(
        model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        device="auto",
        max_steps=8,
        temperature=0.7,
        num_samples=2,
        max_new_tokens=1024,
    )

    problem = mini_bench[4]["question"]

    result, agent_state = agent.solve(problem)

    print_report(result, agent_state)

    console.rule("[bold green]🎯 Session Complete[/bold green]")
