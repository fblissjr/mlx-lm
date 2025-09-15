# Copyright © 2023-2024 Apple Inc.

import argparse

import mlx.core as mx

# The core generation logic is now imported from generate.py
from .generate import stream_generate
from .models.cache import make_prompt_cache
from .sample_utils import make_sampler
from .utils import load

DEFAULT_TEMP = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_XTC_PROBABILITY = 0.0
DEFAULT_XTC_THRESHOLD = 0.0
DEFAULT_SEED = None
DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="Chat with an LLM")
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the local model directory or Hugging Face repo.",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_TOP_P, help="Sampling top-p"
    )
    parser.add_argument(
        "--xtc-probability",
        type=float,
        default=DEFAULT_XTC_PROBABILITY,
        help="Probability of XTC sampling to happen each next token",
    )
    parser.add_argument(
        "--xtc-threshold",
        type=float,
        default=0.0,
        help="Thresold the probs of each next token candidate to be sampled by XTC",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="PRNG seed",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        help="Set the maximum key-value cache size",
        default=None,
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt to be used for the chat template",
    )

    # ADDED: New arguments for speculative decoding and cascades, mirroring generate.py
    parser.add_argument(
        "--draft-model",
        type=str,
        help="A model to be used for speculative decoding or speculative cascades.",
        default=None,
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        help="Number of tokens to draft when using speculative methods.",
        default=3,
    )
    parser.add_argument(
        "--cascade-rule",
        type=str,
        default=None,
        choices=["chow", "diff", "opt"],
        help="Enable speculative cascading with the specified deferral rule. Requires --draft-model.",
    )
    parser.add_argument(
        "--cascade-alpha",
        type=float,
        default=0.1,
        help="The alpha threshold ('strictness') for the speculative cascade rule.",
    )
    return parser


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    # ADDED: Validation to ensure draft model is provided when a cascade rule is specified.
    if args.cascade_rule and not args.draft_model:
        raise ValueError("--cascade-rule requires a --draft-model to be provided.")

    if args.seed is not None:
        mx.random.seed(args.seed)

    # MODIFIED: Load both main model and draft model if provided
    model, tokenizer = load(
        args.model,
        adapter_path=args.adapter_path,
        tokenizer_config={
            "trust_remote_code": True if args.trust_remote_code else None
        },
    )

    draft_model = None
    if args.draft_model:
        print(f"[INFO] Loading draft model from {args.draft_model}...")
        draft_model, draft_tokenizer = load(args.draft_model)
        if draft_tokenizer.vocab_size != tokenizer.vocab_size:
            raise ValueError(
                "The vocabulary of the draft model and main model must be the same."
            )

    def print_help():
        print("The command list:")
        print("- 'q' to exit")
        print("- 'r' to reset the chat")
        print("- 'h' to display these commands")

    print(f"[INFO] Starting chat session with {args.model}.")
    if args.draft_model:
        if args.cascade_rule:
            print(f"[INFO] Using speculative cascade with rule '{args.cascade_rule}' and alpha={args.cascade_alpha}.")
        else:
            print("[INFO] Using standard speculative decoding.")
    print_help()
    prompt_cache = make_prompt_cache(model, args.max_kv_size)

    # Add draft model cache if it exists
    if draft_model:
        prompt_cache.extend(make_prompt_cache(draft_model, args.max_kv_size))

    while True:
        query = input(">> ")
        if query == "q":
            break
        if query == "r":
            # Reset caches for both models if a draft model is in use
            prompt_cache = make_prompt_cache(model, args.max_kv_size)
            if draft_model:
                prompt_cache.extend(make_prompt_cache(draft_model, args.max_kv_size))
            print("[INFO] Chat history reset.")
            continue
        if query == "h":
            print_help()
            continue

        messages = []
        if args.system_prompt is not None:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": query})

        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        # MODIFIED: Pass all new and existing speculative arguments to stream_generate
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            sampler=make_sampler(
                args.temp,
                args.top_p,
                xtc_threshold=args.xtc_threshold,
                xtc_probability=args.xtc_probability,
                xtc_special_tokens=(
                    tokenizer.encode("\n") + list(tokenizer.eos_token_ids)
                ),
            ),
            prompt_cache=prompt_cache,
            draft_model=draft_model,
            num_draft_tokens=args.num_draft_tokens,
            cascade_rule=args.cascade_rule,
            cascade_alpha=args.cascade_alpha,
        ):
            print(response.text, flush=True, end="")
        print()


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.chat...` directly is deprecated."
        " Use `mlx_lm.chat...` or `python -m mlx_lm chat ...` instead."
    )
    main()
