import subprocess
import sys
import os
from pathlib import Path
from termcolor import colored


def run_example_scripts():
    examples_dir = Path(__file__).parent
    this_file = Path(__file__).name
    example_files = [
        f
        for f in examples_dir.glob("*.py")
        if f.name != this_file and f.name != "__init__.py"
    ]

    if not example_files:
        print(colored("No example scripts found.", "yellow"))
        return

    results = []
    print(colored(f"Running {len(example_files)} example scripts...\n", "cyan"))
    for example in sorted(example_files):
        print(colored(f"Running {example.name}...", "blue"))
        try:
            # Set environment variable to enable opening plots in browser
            env = os.environ.copy()
            env["BWR_PLOTS_OPEN_BROWSER"] = "1"

            proc = subprocess.run(
                [sys.executable, str(example)],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            if proc.returncode == 0:
                print(colored(f"SUCCESS: {example.name}", "green"))
                results.append((example.name, True, proc.stdout, proc.stderr))
            else:
                print(colored(f"FAIL: {example.name}", "red"))
                print(colored(proc.stderr, "red"))
                results.append((example.name, False, proc.stdout, proc.stderr))
        except Exception as e:
            print(colored(f"ERROR running {example.name}: {e}", "red"))
            results.append((example.name, False, "", str(e)))
        print("-" * 40)

    # Summary
    n_pass = sum(1 for r in results if r[1])
    n_fail = len(results) - n_pass
    print(colored(f"\nSummary:", "cyan"))
    print(colored(f"  Passed: {n_pass}", "green"))
    print(colored(f"  Failed: {n_fail}", "red" if n_fail else "green"))
    if n_fail:
        print(colored("\nFailed scripts:", "red"))
        for name, ok, _, err in results:
            if not ok:
                print(colored(f"- {name}", "red"))


if __name__ == "__main__":
    run_example_scripts()
