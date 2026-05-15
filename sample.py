"""
Sample new images from a pre-trained DiT.
"""
import random
import os
import torch
from tqdm import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import pdb
from PIL import ImageFile
from DistributionMatching import *
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch

def sample_random_label(class_labels, y):
    y-y.item()
    valid_labels = [label for label in class_labels if label != y]
    return random.choice(valid_labels)

def main(args):
    # Setup PyTorch
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # Load full class index list
    with open('./misc/class_indices.txt', 'r') as fp:
        all_classes = fp.readlines()
    all_classes = [class_index.strip() for class_index in all_classes]

    # Select class subset file based on specification
    spec_to_file = {
        'woof': './misc/class_woof.txt',
        'nette': './misc/class_nette.txt',
        'part1': './misc/class_part1.txt',
        'part2': './misc/class_part2.txt',
        'part3': './misc/class_part3.txt',
        'part4': './misc/class_part4.txt',
        '100': './misc/class100.txt',
        'A': './misc/imagenet-a.txt',
        'B': './misc/imagenet-b.txt',
        'C': './misc/imagenet-c.txt',
        'D': './misc/imagenet-d.txt',
        'E': './misc/imagenet-e.txt',
    }
    class_list_file = spec_to_file.get(args.spec, './misc/class_indices.txt')

    with open(class_list_file, 'r') as fp:
        selected_classes = fp.readlines()

    # Slice class range based on phase
    phase = max(0, args.phase)
    class_start = args.nclass * phase
    class_end = args.nclass * (phase + 1)
    selected_classes = selected_classes[class_start:class_end]
    selected_classes = [selected_class.strip() for selected_class in selected_classes]

    class_indices = []
    for selected_class in selected_classes:
        class_indices.append(all_classes.index(selected_class))

    # Validate auto-download conditions
    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load DiT model
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)

    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Setup diffusion, VAE
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path="pretrained_models/"
    ).to(device)

    batch_size = 1

    # Select guidance loss function for distribution matching
    if args.guidance_loss == 'sinkhorn':
        guidance_loss_fn = compute_ot_distribution_matching_loss
    else:
        guidance_loss_fn = None

    # Load approximate distribution (K-means centroids per class, Eq. 16)
    approximate_distribution = compute_approximate_distribution(
        spec=args.spec,data_path=args.data_path, nclass=args.nclass, phase=args.phase, K_num=args.num_centroids
    )

    print(len(selected_classes))

    for class_index, selected_class in zip(class_indices, selected_classes):
        os.makedirs(os.path.join(args.save_dir, selected_class), exist_ok=True)
        memory_samples = []

        for shift in tqdm(range(args.num_samples // batch_size)):
            # Create sampling noise
            z = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
            y = torch.tensor([class_index], device=device)

            # Generate random label from other classes (Sec. 4.2)
            random_label = torch.tensor(
                sample_random_label(class_indices, y), device=device
            )

            # Setup classifier-free guidance: concatenate conditional and unconditional
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * batch_size, device=device)
            y = torch.cat([y, y_null], 0)

            model_kwargs = dict(
                y=y,
                cfg_scale=args.cfg_scale,
                random_label=random_label,
                # Semantic matching hyperparameters 
                t_upper=args.t_upper,
                t_lower=args.t_lower,
                noise_weight=args.noise_weight,
                label_weight=args.label_weight,
                # Distribution matching hyperparameters
                guidance_scale=args.guidance_scale,
                epsilon=args.ot_epsilon,
                num_iterations=args.ot_num_iterations,
                t_guidance_lower=args.t_guidance_lower,
                t_guidance_upper=args.t_guidance_upper,
            )

            # Sample images with distribution matching guidance
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg,
                z.shape,
                z,
                clip_denoised=False,
                cond_fn=guidance_loss_fn,
                model_kwargs=model_kwargs,
                progress=False,
                approximate_distribution=approximate_distribution,
                memory_samples=memory_samples,
                device=device
            )

            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

            # Update memory buffer 
            memory_samples.extend(samples.detach().split(1))

            # Decode latents to pixel space
            samples = vae.decode(samples / 0.18215).sample

            # Save generated images
            for image_index, image in enumerate(samples):
                save_image(
                    image,
                    os.path.join(
                        args.save_dir, selected_class,
                        f"{image_index + shift * batch_size + args.total_shift}.png"
                    ),
                    normalize=True,
                    value_range=(-1, 1)
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model configuration
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")

    # Sampling configuration
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=100,
                        help="The desired IPC for generation.")
    parser.add_argument("--total-shift", type=int, default=0,
                        help="Index offset for the file name.")

    # Class subset configuration
    parser.add_argument("--spec", type=str, default='none',
                        help="Specific subset for generation.")
    parser.add_argument("--nclass", type=int, default=10,
                        help="The class number for generation.")
    parser.add_argument("--phase", type=int, default=0,
                        help="The phase number for generating large datasets.")
    parser.add_argument("--data_path", type=str, default="/data1/wqc/project/MinimaxDiffusion-main/MinimaxDiffusion-main/data/dataset/train",
                        help="The path of target dataset.")

    # Distribution matching configuration 
    parser.add_argument("--guidance-loss", type=str, default='sinkhorn',
                        choices=['sinkhorn',  'none'],
                        help="Guidance loss type for distribution matching.")
    parser.add_argument("--guidance-scale", type=float, default=0.5,
                        help="Guidance scale for semantic matching loss.")
    parser.add_argument("--num-centroids", type=int, default=10,
                        help="Number of K-means centroids K for approximate distribution (Eq. 16).")
    parser.add_argument("--ot-epsilon", type=float, default=0.1,
                        help="Entropic regularization coefficient for Sinkhorn iteration.")
    parser.add_argument("--ot-num-iterations", type=int, default=5,
                        help="Number of Sinkhorn iterations for OT computation.")
    parser.add_argument("--t-guidance-lower", type=int, default=30,
                        help="Lower timestep threshold for distribution matching guidance.")
    parser.add_argument("--t-guidance-upper", type=int, default=45,
                        help="Upper timestep threshold for distribution matching guidance.")
    # Semantic matching configuration 
    parser.add_argument("--t-upper", type=int, default=900,
                        help="Upper timestep threshold for random label conditioning.")
    parser.add_argument("--t-lower", type=int, default=500,
                        help="Lower timestep threshold for random label conditioning.")
    parser.add_argument("--noise-weight", type=float, default=0.06,
                        help="Scaling factor for injected random noise in conditioning.")
    parser.add_argument("--label-weight", type=float, default=0.01,
                        help="Scaling factor for random label signal in conditioning.")
    
    # Output configuration
    parser.add_argument("--save-dir", type=str, default='../logs/test',
                        help="The directory to put the generated images.")

    args = parser.parse_args()
    main(args)