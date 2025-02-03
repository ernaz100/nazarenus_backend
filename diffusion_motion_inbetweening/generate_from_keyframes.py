# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os
import numpy as np
import torch
from .utils.parser_util import cond_synt_args
from .utils.model_util import create_model_and_diffusion, load_saved_model
from .utils import dist_util
from .model.cfg_sampler import ClassifierFreeSampleModel
from .data_loaders.get_data import get_dataset_loader, DatasetConfig
from .data_loaders.humanml.scripts.motion_process import recover_from_ric
from .data_loaders.humanml.utils.plotting import plot_conditional_samples
import json
import random
from types import SimpleNamespace
from .visualize import vis_utils

args = SimpleNamespace(abs_3d=True, action_file='', action_name='', adam_beta2=0.999, apply_zero_mask=False, arch='unet', augment_type='none', avg_model_beta=0.9999, batch_size=1, clip_range=6.0, cond_mask_prob=0.1, cuda=False, cutoff_point=0, data_dir='', dataset='humanml', device=0, diffusion_steps=1000, dim_mults=[2, 2, 2, 2], drop_redundant=False, edit_mode='benchmark_sparse', editable_features='pos_rot_vel', emb_trans_dec=False, eval_batch_size=32, eval_during_training=False, eval_num_samples=1000, eval_rep_times=3, eval_split='test', ff_size=1024, grad_clip=1.0, gradient_schedule=None, guidance_param=2.5, imputate=False, input_text='', keyframe_conditioned=True, keyframe_guidance_param=1.0, keyframe_mask_prob=0.1, keyframe_selection_scheme='random_joints', lambda_fc=0.0, lambda_rcxyz=0.0, lambda_vel=0.0, latent_dim=512, layers=8, log_interval=1000, lr=0.0001, lr_anneal_steps=0,
model_path = "diffusion_motion_inbetweening/save/condmdi_randomframes/model000750000.pt", motion_length=11.2, motion_length_cut=6.0, motion_path='/Users/ericnazarenus/Desktop/dragbased/diffusion-motion-inbetweening/dataset/HumanML3D/new_joint_vecs_abs_3d/000000.npy', n_keyframes=5, no_text=False, noise_schedule='cosine', num_frames=224, num_repetitions=1, num_samples=1, num_steps=3000000, out_mult=False, output_dir='', overwrite=False, predict_xstart=True, random_proj_scale=10.0, reconstruction_guidance=False, reconstruction_weight=5.0, replacement_distribution='conditional', resume_checkpoint='', save_dir='save/snjua2bq', save_interval=50000, seed=10, sigma_small=True, std_scale_shift=[1.0, 0.0], stop_imputation_at=0, stop_recguidance_at=0, text_condition='', text_prompt='a person throws a ball', time_weighted_loss=False, train_platform_type='NoPlatform', train_x0_as_eps=False, traj_extra_weight=1.0, traj_only=False, transition_length=5, unconstrained=False, unet_adagn=True, unet_zero=True, use_ddim=False, use_fixed_dataset=False, use_fixed_subset=False, use_fp16=True, use_random_proj=False, weight_decay=0.01, xz_only=False, zero_keyframe_loss=False)


def load_and_transform_motion(motion, mean, std):
    """Load and transform a specific motion file for conditional synthesis.
    
    Args:
        motion_path (str): Path to the motion .npy file
        args: Command line arguments
        mean (np.ndarray): Mean values for normalization
        std (np.ndarray): Standard deviation values for normalization
    
    Returns:
        tuple: (transformed_motion, motion_length)
    """
   # motion = np.load("/Users/ericnazarenus/Desktop/dragbased/backend/diffusion_motion_inbetweening/dataset/HumanML3D/new_joint_vecs_abs_3d/000404.npy")

    # Ensure motion length is valid
    min_motion_len = 40
    if len(motion) < min_motion_len:
        raise ValueError(f"Motion length {len(motion)} is too short. Must be at least {min_motion_len} frames.")
    
    # Ensure motion length doesn't exceed maximum
    max_motion_len = 196  # Maximum frames supported
    if len(motion) > max_motion_len:
        motion = motion[:max_motion_len]
    
    m_length = len(motion)
    
    # No need for random selection since we want to keep the original motion
    motion = motion[:m_length]
    
    # Z Normalization
    motion = (motion - mean) / std
    
    # Reshape to match dataloader format: [batch_size, njoints, nfeats, nframes]
    motion = torch.from_numpy(motion).float()
    motion = motion.permute(1, 0).unsqueeze(0).unsqueeze(2)  # [1, njoints, 1, nframes]
    
    # Pad with zeros if needed to reach 196 frames
    if m_length < 196:
        padding = torch.zeros((motion.shape[0], motion.shape[1], motion.shape[2], 196 - m_length))
        motion = torch.cat([motion, padding], dim=3)
    
    return motion, m_length


def generate_inbetween_motion(motion, keyframeIndices,first_keyframe_index = None, motion_editing = False, number_diffusion_steps = 10):
    out_path = "./keyframe_gen"
    max_frames = 196
    use_test_set_prompts = False
    texts = ['']
    print('Loading dataset...')
    split = 'test'
    data = load_dataset(args, max_frames, split=split)
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data, number_diffusion_steps)

    ###################################
    # LOADING THE MODEL FROM CHECKPOINT
    print(f"Loading checkpoints from [{args.model_path}]...")
    load_saved_model(model, args.model_path) 
    model = ClassifierFreeSampleModel(model)  # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking
    ###################################

    mean = np.load("./diffusion_motion_inbetweening/dataset/HumanML3D/Mean_abs_3d.npy")
    std = np.load("./diffusion_motion_inbetweening/dataset/HumanML3D/Std_abs_3d.npy")
    
    # Load specific motion file instead of using dataloader
    input_motions, input_lengths = load_and_transform_motion(motion, mean, std)
    input_lengths = torch.tensor([input_lengths])
    # Ensure input_motions has a batch dimension
    if input_motions.dim() == 3:
        input_motions = input_motions.unsqueeze(0)  # Add batch dimension
    
    # Set model_kwargs for the loaded motion
    model_kwargs = {
        'obs_x0': input_motions,
        'y': {
            'lengths': input_lengths,
            'text': texts if not use_test_set_prompts else model_kwargs['y']['text'],
            'diffusion_steps': args.diffusion_steps
        }
    }

    # Create mask based on keyframeIndices
    batch_size, n_joints, n_feats, n_frames = input_motions.shape
    obs_mask = torch.zeros((batch_size, n_joints, n_feats, n_frames), dtype=torch.bool, device=input_motions.device)
    # Set all joints to True for each keyframe
    for frame_idx in keyframeIndices:
        if motion_editing and first_keyframe_index is not None:
            obs_mask[..., 0:first_keyframe_index-20] = True  # From start to keyframe-20
            obs_mask[..., first_keyframe_index] = True  # At Keyframe
            obs_mask[..., first_keyframe_index+20:] = True   # From keyframe+20 to end
        else:
            obs_mask[...,:193,:, frame_idx] = True # All features except velocities and foot contact
    obs_joint_mask = obs_mask.clone()
   
    input_motions = input_motions.to(dist_util.dev()) # [nsamples, njoints=263, nfeats=1, nframes=196]
    model_kwargs['obs_mask'] = obs_mask
    assert max_frames == input_motions.shape[-1]

    # Arguments
    model_kwargs['y']['text'] = texts
    model_kwargs['y']['diffusion_steps'] = args.diffusion_steps

    all_motions = []
    all_lengths = []
    all_text = []
    all_observed_motions = []
    all_observed_masks = []

    for rep_i in range(args.num_repetitions):
        print(f'### Start sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
        if args.guidance_param != 1:
            # text classifier-free guidance
            model_kwargs['y']['text_scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        if args.keyframe_guidance_param != 1:
            # keyframe classifier-free guidance
            model_kwargs['y']['keyframe_scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.keyframe_guidance_param

        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(
                model,
                (args.batch_size, model.njoints, model.nfeats, max_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            ) # [nsamples, njoints, nfeats, nframes]

        n_joints = 22 if (sample.shape[1] in [263, 264]) else 21
        sample = sample.cpu().permute(0, 2, 3, 1)
        sample = data.dataset.t2m_dataset.inv_transform(sample).float()
        sample = recover_from_ric(sample, n_joints, abs_3d=args.abs_3d)
        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1) # batch_size, n_joints=22, 3, n_frames

        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        if args.unconstrained:
            all_text += ['unconstrained'] * args.num_samples
        else:
            text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
            all_text += model_kwargs['y'][text_key]

        print(f"created {len(all_motions) * args.batch_size} samples")
        # Sampling is done!

    input_motions = input_motions.cpu().permute(0, 2, 3, 1)
    input_motions = data.dataset.t2m_dataset.inv_transform(input_motions).float()
    input_motions = recover_from_ric(data=input_motions, joints_num=n_joints, abs_3d=args.abs_3d)
    input_motions = input_motions.view(-1, *input_motions.shape[2:]).permute(0, 2, 3, 1)
    input_motions = input_motions.cpu().numpy()

    all_motions = np.stack(all_motions) # [num_rep, num_samples, 22, 3, n_frames]
    all_text = np.stack(all_text) # [num_rep, num_samples]
    all_lengths = np.stack(all_lengths) # [num_rep, num_samples]
    return all_motions

def load_dataset(args, max_frames, split='test'):
    conf = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=max_frames,
        split=split,
        hml_mode='train',  # in train mode, you get both text and motion.
        use_abs3d=args.abs_3d,
        traject_only=args.traj_only,
        use_random_projection=args.use_random_proj,
        random_projection_scale=args.random_proj_scale,
        augment_type='none',
        std_scale_shift=args.std_scale_shift,
        drop_redundant=args.drop_redundant,
    )
    data = get_dataset_loader(conf)
    return data


def test()    :
    motion_data = np.load("/Users/ericnazarenus/Desktop/dragbased/backend/diffusion_motion_inbetweening/dataset/HumanML3D/new_joint_vecs_abs_3d/000004.npy")
    print(f"Motion data shape: {motion_data.shape}")
