import argparse
def inf_parse_args() -> argparse.Namespace:
    """ This function parses the arguments passed to the script.

    Returns:
        argparse.Namespace: Namespace containing the arguments.
    """
    
    parser = argparse.ArgumentParser(description="Multimodal Garment Designer argparse.")

    # Model parameters
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-inpainting",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )

    # Input image paths
    parser.add_argument(
        "--person_image_path",
        type=str,
        required=True,
        help="Path to the person image.",
    )
    parser.add_argument(
        "--garment_image_path",
        type=str,
        required=True,
        help="Path to the garment image.",
    )

    # Output directory
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions will be written.",
    )

    # Accelerator parameters
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16).",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers for memory-efficient attention.",
    )

    # Disentangled pipeline parameters
    parser.add_argument(
        "--disentagle",
        action="store_true",
        help="Use the disentangled pipeline.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Text guidance scale.",
    )
    parser.add_argument(
        "--guidance_scale_pose",
        type=float,
        default=7.5,
        help="Pose guidance scale.",
    )
    parser.add_argument(
        "--guidance_scale_sketch",
        type=float,
        default=7.5,
        help="Sketch guidance scale.",
    )
    parser.add_argument(
        "--sketch_cond_rate",
        type=float,
        default=0.2,
        help="Sketch conditioning rate.",
    )
    parser.add_argument(
        "--start_cond_rate",
        type=float,
        default=0.0,
        help="Offset sketch conditioning rate.",
    )

    # Miscellaneous parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="A seed for reproducible results.",
    )

    args = parser.parse_args()
    return args