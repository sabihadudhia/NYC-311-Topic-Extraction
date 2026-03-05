import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run NLP pipeline scripts in order and save all terminal outputs into one text file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pipeline_terminal_output.txt",
        help="Output text filename (default: pipeline_terminal_output.txt)",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use (default: current interpreter)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining scripts even if one fails.",
    )
    return parser.parse_args()


def run_script(script_path: Path, python_executable: str):
    command = [python_executable, str(script_path)]
    started_at = datetime.now()

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=str(script_path.parent),
    )

    finished_at = datetime.now()
    duration = (finished_at - started_at).total_seconds()

    combined_output = ""
    if result.stdout:
        combined_output += "[STDOUT]\n" + result.stdout
    if result.stderr:
        if combined_output and not combined_output.endswith("\n"):
            combined_output += "\n"
        combined_output += "[STDERR]\n" + result.stderr

    return {
        "script": str(script_path),
        "command": " ".join(command),
        "return_code": result.returncode,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": duration,
        "output": combined_output if combined_output else "[NO OUTPUT]\n",
    }


def build_report(run_results):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("=" * 100)
    lines.append("NLP PIPELINE COMBINED TERMINAL OUTPUT")
    lines.append("=" * 100)
    lines.append(f"Generated at: {timestamp}")
    lines.append("")

    lines.append("Execution Summary:")
    for idx, item in enumerate(run_results, start=1):
        status = "SUCCESS" if item["return_code"] == 0 else "FAILED"
        lines.append(
            f"{idx}. {Path(item['script']).name} -> {status} "
            f"(exit={item['return_code']}, duration={item['duration_seconds']:.2f}s)"
        )
    lines.append("")

    for idx, item in enumerate(run_results, start=1):
        lines.append("-" * 100)
        lines.append(f"STEP {idx}: {Path(item['script']).name}")
        lines.append("-" * 100)
        lines.append(f"Script Path : {item['script']}")
        lines.append(f"Command     : {item['command']}")
        lines.append(f"Started At  : {item['started_at'].strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Finished At : {item['finished_at'].strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Duration    : {item['duration_seconds']:.2f} seconds")
        lines.append(f"Exit Code   : {item['return_code']}")
        lines.append("")
        lines.append("Output:")
        lines.append(item["output"].rstrip("\n"))
        lines.append("")

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()

    project_dir = Path(__file__).resolve().parent
    scripts_in_order = [
        project_dir / "data_preprocessing.py",
        project_dir / "text_vectorization.py",
        project_dir / "topic_extraction.py",
        project_dir / "advanced_topic_analysis.py",
    ]

    missing = [str(path) for path in scripts_in_order if not path.exists()]
    if missing:
        print("Error: The following required scripts were not found:")
        for path in missing:
            print(f"  - {path}")
        sys.exit(1)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_dir / output_path

    print("Running pipeline scripts in order...")
    print(f"Python executable: {args.python}")
    print(f"Output file: {output_path}")

    run_results = []

    for script in scripts_in_order:
        print(f"\nRunning: {script.name}")
        result = run_script(script, args.python)
        run_results.append(result)

        if result["return_code"] == 0:
            print(f"  Completed successfully in {result['duration_seconds']:.2f}s")
        else:
            print(f"  Failed with exit code {result['return_code']}")
            if not args.continue_on_error:
                print("  Stopping pipeline due to failure. Use --continue-on-error to run all scripts.")
                break

    report = build_report(run_results)
    output_path.write_text(report, encoding="utf-8")

    print("\n" + "=" * 100)
    print(f"Combined output saved to: {output_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
