# python inf.py --person_image_path ./dresses/model.jpg --garment_image_path ./dresses/KIZ_1388.jpg --output_dir ./output
#  external libraries
import os
import torch
from diffusers import UNet2DConditionModel

from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer

# custom imports

from mgd_pipelines.mgd_pipe import MGDPipe
from mgd_pipelines.mgd_pipe_disentangled import MGDPipeDisentangled
from utils.inf_parser import inf_parse_args
from utils.set_seeds import set_seed

from PIL import Image
from torchvision import transforms

logger = get_logger(__name__, log_level="INFO")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB_START_METHOD"] = "thread"

def mgd(model: str = 'vitonhd', **kwargs) -> UNet2DConditionModel:


    config = UNet2DConditionModel.load_config("runwayml/stable-diffusion-inpainting", subfolder="unet")
    config['in_channels'] = 28
    unet = UNet2DConditionModel.from_config(config)

    model = f"./models/{model}.pth"
    if not os.path.exists(model):
        raise ValueError(f"Model file {model} does not exist. Please download the model first.")
    
    unet.load_state_dict(torch.load(model))

    return unet


def main() -> None:
    args = inf_parse_args()
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )
    device = accelerator.device

    # Set the seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)

    # Load scheduler, tokenizer, and models
    val_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    val_scheduler.set_timesteps(50, device=device)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)

    unet = mgd()

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Enable memory efficient attention if requested
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.category:
        category = [args.category]
    else:
        category = ['dresses', 'upper_body', 'lower_body']

    if args.dataset == "dresscode":
        test_dataset = DressCodeDataset(
            dataroot_path=args.dataset_path,
            phase='test',
            order=args.test_order,
            radius=5,
            sketch_threshold_range=(20, 20),
            tokenizer=tokenizer,
            category=category,
            size=(512, 384)
        )
    elif args.dataset == "vitonhd":
        test_dataset = VitonHDDataset(
            dataroot_path=args.dataset_path,
            phase='test',
            order=args.test_order,
            sketch_threshold_range=(20, 20),
            radius=5,
            tokenizer=tokenizer,
            size=(512, 384),
        )
    else:
        raise NotImplementedError

    # Set precision
    weight_dtype = torch.float32
    if args.mixed_precision == 'fp16':
        weight_dtype = torch.float16

    # Move models to device
    text_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)

    unet.eval()
    with torch.inference_mode():
        if args.disentagle:
            val_pipe = MGDPipeDisentangled(
                text_encoder=text_encoder,
                vae=vae,
                unet=unet.to(vae.dtype),
                tokenizer=tokenizer,
                scheduler=val_scheduler,
            ).to(device)
        else:
            val_pipe = MGDPipe(
                text_encoder=text_encoder,
                vae=vae,
                unet=unet.to(vae.dtype),
                tokenizer=tokenizer,
                scheduler=val_scheduler,
            ).to(device)

        val_pipe.enable_attention_slicing()

        # Load and preprocess the person and garment images
        person_image_path = args.person_image_path  # Path to the person image
        garment_image_path = args.garment_image_path  # Path to the garment image

        # Preprocess the person and garment images
        preprocess_person = transforms.Compose([
            transforms.Resize((512, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Normalize to match model input
        ])

        preprocess_garment = transforms.Compose([
            transforms.Resize((512, 384)),
            transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
            transforms.Grayscale(num_output_channels=1),  # Convert to single-channel grayscale
        ])

        person_image = preprocess_person(Image.open(person_image_path).convert("RGB")).unsqueeze(0).to(device)
        garment_image = preprocess_garment(Image.open(garment_image_path).convert("RGB")).unsqueeze(0).to(device)

        # Generate the output
        if args.disentagle:    
            output = val_pipe(
                prompt="image of girl wearing a dress",  # Replace with the actual prompt string or list
                image=person_image,  # Map person_image to the 'image' argument
                mask_image=garment_image,  # Map garment_image to the 'mask_image' argument
                pose_map=torch.zeros_like(person_image),  # Provide a placeholder pose map if not available
                sketch=torch.zeros_like(person_image),  # Provide a placeholder sketch if not available
                guidance_scale= 7.5,
                guidance_scale_pose= 7.5,
                guidance_scale_sketch= 7.5,
                sketch_cond_rate= 1.0,
                start_cond_rate= 0.0,
            )
        else:
        # Generate the output
            output = val_pipe(
                prompt="image of girl wearing a dress",  # Replace with the actual prompt string or list
                image=person_image,  # Map person_image to the 'image' argument
                mask_image=garment_image,  # Map garment_image to the 'mask_image' argument
                pose_map=torch.zeros_like(person_image),  # Provide a placeholder pose map if not available
                sketch=torch.zeros_like(person_image),  # Provide a placeholder sketch if not available
                num_inference_steps=50,  # Number of denoising steps
                guidance_scale=7.5,  # Guidance scale for classifier-free guidance
                sketch_cond_rate=1.0,  # Sketch conditioning rate
                start_cond_rate=0.0,  # Start conditioning rate
                output_type="pil",  # Output format (PIL image)
            )

        # Save the generated image
        output_image = output.images[0]
        output_image.save(args.output_dir + "/generated_image.png")
        print(f"Generated image saved to {args.output_dir}/generated_image.png")

if __name__ == "__main__":
    main()