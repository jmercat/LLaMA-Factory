import re
from regex import F
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse


def load_model(base_model_name, weights_path, use_lora=True):
    """
    Load the model with either LoRA weights or full model weights.

    Args:
        base_model_name: Name of the base model
        weights_path: Path to either LoRA weights or full model checkpoint
        use_lora: If True, load as LoRA weights. If False, load as full checkpoint
    """
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(weights_path, trust_remote_code=True)

    if use_lora:
        # Load base model first, then apply LoRA weights
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        # Load and merge LoRA weights
        model = PeftModel.from_pretrained(model, weights_path)
    else:
        # Load full model checkpoint directly
        model = AutoModelForCausalLM.from_pretrained(
            weights_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    return model, tokenizer


def decode_octree(octree):
    """
    Take in a string of octree tokens and decode it into a list of list of integers
    """
    # Remove anything that is not <octree_n> or [SEP]
    octree = octree.split("\n")[2].replace("<|im_end|>", "")
    # Convert <octree_n> to list of n
    octree = octree.replace("<octree_", "").replace(">", ",")
    octree = octree.split("[SEP]")
    octree = [
        list(map(lambda x: int(x.strip()), octree_str.split(",")[:-1]))
        for octree_str in octree
    ]
    return octree


def generate_octree(model, tokenizer, text_prompt, max_length=2048):
    """Generate an octree representation for a given text description."""
    # Create a conversation in the same format used during training
    messages = [
        {
            "role": "user",
            "content": f"Convert this 3D shape description to an octree representation: {text_prompt}",
        }
    ]

    # Format using the chat template that was used during training
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=tokenizer.pad_token_id,
            temperature=0.0,  # Adjust for more/less randomness
            do_sample=False,
            top_p=0.9,
            top_k=50,
        )

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract just the assistant's response
    # This depends on the specific chat template, but we can use a more robust approach
    # by looking for the last user message and taking everything after it
    user_prompt = (
        f"Convert this 3D shape description to an octree representation: {text_prompt}"
    )
    if user_prompt in response:
        octree = response[response.find(user_prompt) + len(user_prompt) :].strip()
    else:
        # Fallback: just return everything after the input prompt
        octree = response[len(prompt) :].strip()

    return decode_octree(octree)


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Generate octrees from text descriptions"
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        required=True,
        help="Path to either LoRA weights or full model checkpoint",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="If set, load as LoRA weights. If not set, load as full checkpoint",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model name (default: Qwen/Qwen2.5-0.5B-Instruct)",
    )

    parser.add_argument(
        "--text",
        type=str,
        default="",
        help="Text description of the 3D shape to generate",
    )

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model, tokenizer = load_model(args.base_model, args.weights_path, args.use_lora)

    if args.text == "":
        # Interactive loop
        print("\nEnter text descriptions (or 'quit' to exit):")
        while True:
            text = input("\nDescription: ").strip()
            if text.lower() == "quit":
                break
            print("Generating octree...")
            octree = generate_octree(model, tokenizer, text)
            print("\nGenerated Octree:")
            print(octree)
    else:
        octree = generate_octree(model, tokenizer, args.text)
        print("\nGenerated Octree:")
        print(octree)

    return octree


if __name__ == "__main__":
    main()
