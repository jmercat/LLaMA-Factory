import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

import numpy as np
import ast
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def dec2bin(x, bits=8):
    mask = 2 ** np.arange(bits)
    return (x[..., None] & mask) != 0


def subdivide(centers, level):
    offset_size = (1 / (2**level)) * 2 / 4
    offsets = (
        np.array(
            [
                (-1, -1, -1),
                (-1, -1, 1),
                (-1, 1, -1),
                (-1, 1, 1),
                (1, -1, -1),
                (1, -1, 1),
                (1, 1, -1),
                (1, 1, 1),
            ]
        )
        * offset_size
    )

    # Ensure centers are correctly expanded
    centers_expanded = np.repeat(centers, 8, axis=-2)  # Expand each center to 8 copies

    # Expand offsets to match shape
    offsets_expanded = np.tile(offsets, (centers.shape[1], 1))[None, ...]
    centers_new = centers_expanded + offsets_expanded

    return centers_new


def visualize_octree(octree, save_path=None):
    """
    Visualize the octree using matplotlib with Poly3DCollection for better voxel rendering.
    """
    # Octree string to list of numpy lists
    octree_list = ast.literal_eval(octree)
    octree_list = [np.array([int(x) for x in lod]) for lod in octree_list]

    # Generate points
    points = [np.zeros((1, 1, 3), dtype=np.float32)]
    for lod in range(len(octree_list)):
        occ = dec2bin(octree_list[lod]).astype(bool).reshape(1, -1)
        xyz = subdivide(points[lod], level=lod)[occ].reshape(1, -1, 3)
        points.append(xyz)

    voxel_size = (1 / (2 ** len(octree_list))) * 2  # Size of each voxel
    voxels = points[-1][0]
    colors = (voxels + 1) / 2  # Normalize colors

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Draw each voxel as a cube with faces
    for voxel, color in zip(voxels, colors):
        x, y, z = voxel
        draw_voxel(ax, x, y, z, voxel_size, color)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect([1, 1, 1])  # Ensures cubic aspect ratio

    # Either save to file or display
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")
    else:
        try:
            plt.show()
        except Exception as e:
            print(f"Failed to display visualization: {e}")
            print(
                "In headless environments, use --output_image to save the visualization to a file."
            )

    plt.close(fig)


def draw_voxel(ax, x, y, z, size, color):
    """
    Draw a voxel as a cube with colored faces using Poly3DCollection.
    """
    r = [-size / 2, size / 2]
    vertices = np.array(
        [
            [x + r[i], y + r[j], z + r[k]]
            for i in range(2)
            for j in range(2)
            for k in range(2)
        ]
    )
    faces = [
        [vertices[j] for j in [0, 1, 3, 2]],
        [vertices[j] for j in [4, 5, 7, 6]],
        [vertices[j] for j in [0, 1, 5, 4]],
        [vertices[j] for j in [2, 3, 7, 6]],
        [vertices[j] for j in [0, 2, 6, 4]],
        [vertices[j] for j in [1, 3, 7, 5]],
    ]

    # Create a collection with the faces
    collection = Poly3DCollection(
        faces, facecolors=[color], alpha=0.7, edgecolor="k", linewidth=0.5
    )
    ax.add_collection3d(collection)


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


def generate_octree(model, tokenizer, text_prompt):
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
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            max_new_tokens=9000,
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

    decoded_octree = decode_octree(octree)
    return decoded_octree


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Generate octrees from text descriptions and visualize them"
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
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the generated octree",
    )
    parser.add_argument(
        "--octree",
        type=str,
        default=None,
        help="Directly visualize this octree string instead of generating one",
    )
    parser.add_argument(
        "--output_image",
        type=str,
        default=None,
        help="Path to save visualization image instead of displaying it",
    )
    parser.add_argument(
        "--max_lod",
        type=int,
        default=6,
        help="Maximum level of detail to generate",
    )

    args = parser.parse_args()

    # If only visualization is needed
    if args.octree is not None:
        visualize_octree(args.octree, args.output_image)
        return

    # Load model for generation
    if args.text != "" or args.text == "" and args.octree is None:
        print("Loading model...")
        model, tokenizer = load_model(args.base_model, args.weights_path, args.use_lora)

        if args.text == "":
            # Interactive loop
            print("\nEnter text descriptions (or 'quit' to exit):")
            while True:
                text = input("\nDescription: ").strip()
                try:
                    if text.lower() == "quit":
                        break
                    print("Generating octree...")
                    octree = generate_octree(model, tokenizer, text)
                    print("\nGenerated Octree:")
                    print(octree)

                    # Visualize if requested
                    success = False
                    lod = args.max_lod
                    while not success and lod > 0:
                        try:
                            octree = octree[: lod + 1]
                            # Visualize if requested
                            if args.visualize:
                                print("Visualizing octree...")
                                if args.output_image:
                                    name = (
                                        text.replace(" ", "_")
                                        .replace(".", "")
                                        .replace(",", "")
                                    )
                                    output_image = os.path.join(
                                        args.output_image, f"{name}.png"
                                    )
                                else:
                                    output_image = None
                                visualize_octree(str(octree), output_image)
                                success = True
                        except Exception as e:
                            print(f"Error at lod {lod}: {e}")
                            success = False
                            lod -= 1
                except Exception as e:
                    print(f"Error: {e}")
        else:
            octree = generate_octree(model, tokenizer, args.text)
            print("\nGenerated Octree:")
            print(octree)

            success = False
            lod = args.max_lod
            while not success and lod > 0:
                try:
                    octree = octree[: lod + 1]
                    # Visualize if requested
                    if args.visualize:
                        print("Visualizing octree...")
                    if args.output_image:
                        name = (
                            args.text.replace(" ", "_")
                            .replace(".", "")
                            .replace(",", "")
                        )
                        output_image = os.path.join(args.output_image, f"{name}.png")
                    else:
                        output_image = None
                        visualize_octree(str(octree), output_image)
                        success = True
                except Exception as e:
                    print(f"Error at lod {lod}: {e}")
                    success = False
                    lod -= 1


if __name__ == "__main__":
    main()
