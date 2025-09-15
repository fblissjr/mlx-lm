import argparse
import json
import re
import subprocess
import time
import datetime

# script to benchmark speculative cascades against baselines - runs mlx_lm.generate as a subprocess to ensure clean memory state for each run

DEFAULT_PROMPT = "Write a detailed three-paragraph short story about an astronaut who rides a horse on the moon."
DEFAULT_MAX_TOKENS = 256
DEFAULT_NUM_DRAFT_TOKENS = 3

# speculative cascade alpha values to sweep
ALPHA_SWEEP = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]

def run_benchmark_command(command: list[str]) -> dict:
    """Runs a command and parses the verbose output to extract performance metrics."""
    print("\n" + "="*80)
    print(f"Running command: {' '.join(command)}")
    print("="*80)

    start_time = time.time()
    try:
        # prevent hanging on a broken implementation
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
            timeout=1800,
        )
        # Now stdout contains both the model output and the debug messages
        full_output = result.stdout

        # generated output and any debug info
        print("\n--- Model Generation & Debug Output (stdout) ---")
        print(full_output)
        print("------------------------------------------------\n")


        # parse performance metrics from stdout
        lines = full_output.strip().split('\n')
        gen_line = next((line for line in reversed(lines) if "Generation" in line), None)
        mem_line = next((line for line in reversed(lines) if "Peak memory" in line), None)

        if not gen_line or not mem_line:
            raise ValueError("Could not parse performance metrics from output.")

        # get the main text output, ignoring debug lines
        output_text_lines = []
        in_main_output = False
        for line in lines:
            if "==========" in line:
                in_main_output = not in_main_output # Toggle state
                continue
            if in_main_output and not line.strip().startswith("[DEBUG"):
                output_text_lines.append(line)
        output_text = "\n".join(output_text_lines)

        tokens_per_sec = float(gen_line.split(',')[-2].split(' ')[1])
        peak_memory_gb = float(mem_line.split(' ')[2])

        # parse acceptance rate from the full output
        acceptance_rate = None
        rate_match = re.search(r"\((\d+\.\d+)% acceptance rate\)", full_output)
        if rate_match:
            acceptance_rate = float(rate_match.group(1))

        return {
            "output": output_text,
            "tokens_per_sec": tokens_per_sec,
            "peak_memory_gb": peak_memory_gb,
            "acceptance_rate_percent": acceptance_rate,
            "run_time_sec": time.time() - start_time
        }

    except subprocess.CalledProcessError as e:
        # stderr redirected, e.stdout will contain the full error log
        print(f"ERROR: Command failed with exit code {e.returncode}")
        print("--- COMBINED OUTPUT ---")
        print(e.stdout)
        return {"error": str(e)}
    except subprocess.TimeoutExpired as e:
        print(f"ERROR: Command timed out after {e.timeout} seconds.")
        return {"error": "timeout"}

def main():

    # construct results file with datetime
    results_file = f"cascade_benchmark_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    parser = argparse.ArgumentParser(description="Benchmark script for Speculative Cascades.")
    parser.add_argument(
        "--results-file",
        type=str,
        default=results_file,
        help="File to save the benchmark results."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The verifier model aka the larger model"
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        required=True,
        help="The drafter model aka the smaller model"
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default=DEFAULT_PROMPT,
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate."
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=DEFAULT_NUM_DRAFT_TOKENS,
        help="Number of tokens for the draft model to generate."
    )
    args = parser.parse_args()

    VERIFIER_MODEL = args.model
    DRAFTER_MODEL = args.draft_model
    PROMPT = args.prompt
    MAX_TOKENS = args.max_tokens
    NUM_DRAFT_TOKENS = args.num_draft_tokens

    results = {}

    base_command = ["mlx_lm.generate", "--prompt", PROMPT, "--max-tokens", str(MAX_TOKENS), "--verbose", "True"]

    # baseline 1 - verifier model only
    print("\n\n--- BENCHMARKING: Verifier Model Only ---")
    command = base_command + ["--model", VERIFIER_MODEL]
    results["verifier_only"] = run_benchmark_command(command)

    # baseline 2 -drafter model only
    print("\n\n--- BENCHMARKING: Drafter Model Only ---")
    command = base_command + ["--model", DRAFTER_MODEL]
    results["drafter_only"] = run_benchmark_command(command)

    # baseline 3 - normal speculative Decoding ---
    print("\n\n--- BENCHMARKING: Standard Speculative Decoding ---")
    command = base_command + [
        "--model", VERIFIER_MODEL,
        "--draft-model", DRAFTER_MODEL,
        "--num-draft-tokens", str(NUM_DRAFT_TOKENS)
    ]
    results["standard_speculative"] = run_benchmark_command(command)

    # Speculative Cascade sweep
    results["speculative_cascade_opt"] = {}
    print("\n\n--- BENCHMARKING: Speculative Cascade with 'opt' rule ---")
    for alpha in ALPHA_SWEEP:
        print(f"\n--- Testing alpha = {alpha} ---")
        command = base_command + [
            "--model", VERIFIER_MODEL,
            "--draft-model", DRAFTER_MODEL,
            "--cascade-rule", "opt",
            "--cascade-alpha", str(alpha),
            "--num-draft-tokens", str(NUM_DRAFT_TOKENS)
        ]
        results["speculative_cascade_opt"][f"alpha_{alpha}"] = run_benchmark_command(command)

    # Save results to a file
    with open(args.results_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n\nBenchmark complete. Results saved to {args.results_file}")

    # summary table
    print("\n\n--- Benchmark Summary ---")
    print(f"{'Configuration':<40} | {'Speed (tok/s)':<15} | {'Accept Rate (%)':<17} | {'Peak Memory (GB)':<18}")
    print("-" * 95)

    def print_summary_line(name, result):
        if "error" in result:
            print(f"{name:<40} | {'ERROR':<15} | {'N/A':<17} | {'N/A':<18}")
        else:
            speed = f"{result['tokens_per_sec']:.2f}"
            rate = f"{result['acceptance_rate_percent']:.1f}" if result.get('acceptance_rate_percent') is not None else "N/A"
            mem = f"{result['peak_memory_gb']:.2f}"
            print(f"{name:<40} | {speed:<15} | {rate:<17} | {mem:<18}")

    print_summary_line("Verifier Only", results["verifier_only"])
    print_summary_line("Drafter Only", results["drafter_only"])
    print_summary_line("Standard Speculative Decoding", results["standard_speculative"])
    for alpha_str, result in results["speculative_cascade_opt"].items():
        # alpha_str = "alpha_0.1"
        alpha_val = alpha_str.split('_')[1]
        print_summary_line(f"Cascade (opt, α={alpha_val})", result)


if __name__ == "__main__":
    main()
