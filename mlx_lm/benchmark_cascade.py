import argparse
import json
import re
import subprocess
import time
from pathlib import Path
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
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
            timeout=1800,
        )
        full_output = result.stdout

        # generated output and any debug info
        print("\n--- Model Generation & Debug Output (stdout) ---")
        print(full_output)
        print("------------------------------------------------\n")

        lines = full_output.strip().split('\n')

        # parse performance metrics from stdout
        prompt_metrics = None
        gen_metrics = None
        mem_metrics = None

        prompt_pattern = re.compile(r"Prompt:\s+(\d+)\s+tokens,\s+([\d\.]+)\s+tokens-per-sec")
        gen_pattern = re.compile(r"Generation:\s+(\d+)\s+tokens,\s+([\d\.]+)\s+tokens-per-sec")
        mem_pattern = re.compile(r"Peak memory:\s+([\d\.]+)\s+GB")

        for line in reversed(lines):
            if gen_metrics is None:
                gen_match = gen_pattern.search(line)
                if gen_match:
                    gen_metrics = {
                        "generation_tokens": int(gen_match.group(1)),
                        "generation_tok_per_sec": float(gen_match.group(2))
                    }
                    continue
            if prompt_metrics is None:
                prompt_match = prompt_pattern.search(line)
                if prompt_match:
                    prompt_metrics = {
                        "prompt_tokens": int(prompt_match.group(1)),
                        "prompt_tok_per_sec": float(prompt_match.group(2))
                    }
                    continue
            if mem_metrics is None:
                mem_match = mem_pattern.search(line)
                if mem_match:
                    mem_metrics = {
                        "peak_memory_gb": float(mem_match.group(1))
                    }
                    continue

        if not gen_metrics or not prompt_metrics or not mem_metrics:
            raise ValueError("Could not parse all required performance metrics from output.")

        # get the main text output, ignoring debug lines
        output_text_lines = []
        in_main_output = False
        for line in lines:
            if "==========" in line:
                in_main_output = not in_main_output
                continue
            if in_main_output and not line.strip().startswith("[DEBUG"):
                output_text_lines.append(line)
        output_text = "\n".join(output_text_lines)

        # parse acceptance rate from the full output
        acceptance_rate = None
        rate_match = re.search(r"\((\d+\.\d+)% acceptance rate\)", full_output)
        if rate_match:
            acceptance_rate = float(rate_match.group(1))

        all_metrics = {
            "output": output_text,
            "acceptance_rate_percent": acceptance_rate,
            "run_time_sec": time.time() - start_time,
            **prompt_metrics,
            **gen_metrics,
            **mem_metrics,
        }
        return all_metrics

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with exit code {e.returncode}")
        print("--- COMBINED OUTPUT ---")
        print(e.stdout)
        return {"error": str(e)}
    except subprocess.TimeoutExpired as e:
        print(f"ERROR: Command timed out after {e.timeout} seconds.")
        return {"error": "timeout"}
def main():

    parser = argparse.ArgumentParser(description="Benchmark script for Speculative Cascades.")
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
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
    if args.results_file is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            args.results_file = f"cascade_benchmark_results_{timestamp}.json"

    print(f"\n\nBenchmark complete. Results saved to {args.results_file}")

    # summary table
    print("\n\n" + "="*120)
    print("--- Benchmark Summary ---")
    print("="*120)
    header = f"{'Configuration':<35} | {'Gen Tok/s':<11} | {'Accept Rate (%)':<15} | {'Prompt Tok/s':<12} | {'Gen Tokens':<10} | {'Peak Mem (GB)':<15}"
    print(header)
    print("-" * 120)

    def print_summary_line(name, result):
        if "error" in result:
            print(f"{name:<35} | {'ERROR':<11} | {'N/A':<15} | {'N/A':<12} | {'N/A':<10} | {'N/A':<15}")
        else:
            speed = f"{result['generation_tok_per_sec']:.2f}"
            rate = f"{result['acceptance_rate_percent']:.1f}" if result.get('acceptance_rate_percent') is not None else "N/A"
            prompt_speed = f"{result['prompt_tok_per_sec']:.2f}"
            gen_tokens = f"{result['generation_tokens']}"
            mem = f"{result['peak_memory_gb']:.2f}"
            print(f"{name:<35} | {speed:<11} | {rate:<15} | {prompt_speed:<12} | {gen_tokens:<10} | {mem:<15}")

    print_summary_line("Verifier Only", results["verifier_only"])
    print_summary_line("Drafter Only", results["drafter_only"])
    print_summary_line("Standard Speculative Decoding", results["standard_speculative"])
    for alpha_str, result in results["speculative_cascade_opt"].items():
        alpha_val = alpha_str.split('_')[1]
        print_summary_line(f"Cascade (opt, α={alpha_val})", result)

    print("="*120)


if __name__ == "__main__":
    main()
