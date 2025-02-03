import os
import shutil

import numpy as np
import torch

from priorMDM.data_loaders.get_data import get_dataset_loader
from priorMDM.data_loaders.humanml.scripts.motion_process import recover_from_ric
from priorMDM.data_loaders.humanml.utils import paramUtil
from priorMDM.data_loaders.humanml.utils.plot_script import plot_3d_motion
from priorMDM.data_loaders.humanml_utils import get_inpainting_mask
from priorMDM.diffusion.inpainting_gaussian_diffusion import InpaintingGaussianDiffusion
from priorMDM.diffusion.respace import SpacedDiffusion
from priorMDM.utils import dist_util
from priorMDM.utils.fixseed import fixseed
from priorMDM.utils.model_util import load_model_blending_and_diffusion
from priorMDM.utils.parser_util import edit_inpainting_args


def mini_prior_mdm(humanml3d:np.ndarray, num_diffusion_steps: int):
    # some setup
    args_list = edit_inpainting_args()
    args = args_list[0]
    fixseed(args.seed)
    dist_util.setup_dist(args.device)


    args.diffusion_steps = num_diffusion_steps
    input_motion = torch.tensor(humanml3d).T.unsqueeze(0).unsqueeze(2)
    max_frames = input_motion.shape[-1]


    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    all_motions = []
    all_lengths = []
    all_text = []
    gt_frames_per_sample = {}
    total_num_samples = args.num_samples * args.num_repetitions
    fps = 12.5 if args.dataset == 'kit' else 20

    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'edit_{}_{}_{}_seed{}'.format(name, niter, args.inpainting_mask, args.seed))

    print('Loading dataset...')
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    # df: we only need this for setup. We want to overwrite its behavior
    data = get_dataset_loader(name = args.dataset,
                              batch_size = args.batch_size,
                              num_frames = max_frames,
                              split = 'test',
                              load_mode = 'train',
                              size = args.num_samples)  # in train mode, you get both text and motion.

    print("Creating model and diffusion...")
    DiffusionClass = InpaintingGaussianDiffusion if args_list[0].filter_noise else SpacedDiffusion
    model, diffusion = load_model_blending_and_diffusion(args_list, dist_util.dev(),DiffusionClass=DiffusionClass)

    model_kwargs = {'y': {}}
    model_kwargs['y']['inpainted_motion'] = input_motion
    model_kwargs['y']['inpainting_mask'] = torch.tensor(
        get_inpainting_mask(args.inpainting_mask, input_motion.shape)).float().to(dist_util.dev())
    model_kwargs['y']['text'] = [""]
    model_kwargs['y']['lengths'] = torch.tensor(max_frames, device =dist_util.dev()).unsqueeze(0)

    print(f"Num. repetitions: {args.num_repetitions}")
    print(f"Num. Samples: {args.num_samples}")

    print(f'### Start sampling')

    # add CFG scale to batch
    model_kwargs['y']['scale'] = torch.ones(args.batch_size, device = dist_util.dev()) * args.guidance_param

    sample_fn = diffusion.p_sample_loop

    new_generated_motion = sample_fn(
        model,
        (args.batch_size, model.njoints, model.nfeats, max_frames),
        clip_denoised = False,
        model_kwargs = model_kwargs,
        skip_timesteps = 0,  # 0 is the default value - i.e. don't skip any step
        init_image = input_motion,
        progress = True,
        dump_steps = None,
        noise = None,
        const_noise = False,
    )

    # Recover XYZ *positions* from HumanML3D vector representation
    if model.data_rep == 'hml_vec':
        n_joints = 22 if new_generated_motion.shape[1] == 263 else 21
        new_generated_motion = data.dataset.t2m_dataset.inv_transform(new_generated_motion.cpu().permute(0, 2, 3, 1)).float()
        new_generated_motion = recover_from_ric(new_generated_motion, n_joints)
        new_generated_motion = new_generated_motion.view(-1, *new_generated_motion.shape[2:]).permute(0, 2, 3, 1)

    all_text += model_kwargs['y']['text']
    all_motions.append(new_generated_motion.cpu().numpy())
    all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())


    # The rest is just there to save the animation
    print(f"created {len(all_motions) * args.batch_size} samples")

    all_motions = np.concatenate(all_motions, axis = 0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis = 0)[:total_num_samples]

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

    # Recover XYZ *positions* from HumanML3D vector representation
    if model.data_rep == 'hml_vec':
        input_motions = data.dataset.t2m_dataset.inv_transform(input_motion.cpu().permute(0, 2, 3, 1)).float()
        input_motions = recover_from_ric(input_motions, n_joints)
        input_motions = input_motions.view(-1, *input_motions.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()

    for sample_i in range(args.num_samples):
        rep_files = []
        if args.show_input:
            caption = 'Input Motion'
            length = model_kwargs['y']['lengths'][sample_i]
            motion = input_motions[sample_i].transpose(2, 0, 1)[:length]
            save_file = 'input_motion{:02d}.mp4'.format(sample_i)
            animation_save_path = os.path.join(out_path, save_file)
            rep_files.append(animation_save_path)
            print(f'[({sample_i}) "{caption}" | -> {save_file}]')
            plot_3d_motion(animation_save_path, skeleton, motion, title = caption,
                           dataset = args.dataset, fps = fps, vis_mode = 'gt',
                           gt_frames = gt_frames_per_sample.get(sample_i, []))
        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i * args.batch_size + sample_i]
            if args.guidance_param == 0:
                caption = 'Edit [{}] unconditioned'.format(args.inpainting_mask)
            else:
                caption = 'Edit [{}]: {}'.format(args.inpainting_mask, caption)
            length = all_lengths[rep_i * args.batch_size + sample_i]
            motion = all_motions[rep_i * args.batch_size + sample_i].transpose(2, 0, 1)[:length]
            save_file = 'sample{:02d}_rep{:02d}.mp4'.format(sample_i, rep_i)
            animation_save_path = os.path.join(out_path, save_file)
            rep_files.append(animation_save_path)
            print(f'[({sample_i}) "{caption}" | Rep #{rep_i} | -> {animation_save_path}]')
            plot_3d_motion(animation_save_path, skeleton, motion, title = caption,
                           dataset = args.dataset, fps = fps, vis_mode = args.inpainting_mask,
                           gt_frames = gt_frames_per_sample.get(sample_i, []))
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion

        if args.num_repetitions > 1:
            all_rep_save_file = os.path.join(out_path, 'sample{:02d}.mp4'.format(sample_i))
            ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
            hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions + (1 if args.show_input else 0)} '
            ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(
                ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_file}'
            os.system(ffmpeg_rep_cmd)
            print(f'[({sample_i}) "{caption}" | all repetitions | -> {all_rep_save_file}]')

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')

    # Bring the new generated motion into (22, 3, sequence_length) shape
    return new_generated_motion.squeeze()
