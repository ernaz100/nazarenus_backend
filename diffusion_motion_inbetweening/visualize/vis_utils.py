from ..model.rotation2xyz import Rotation2xyz
import numpy as np
import trimesh
from trimesh import Trimesh
import os
import torch
from .simplify_loc2rot import joints2smpl

class npy2obj:
    def __init__(self, npy_path, sample_idx, rep_idx, device=0, cuda=True):
        self.npy_path = npy_path
        self.motions = np.load(self.npy_path, allow_pickle=True)
        if self.npy_path.endswith('.npz'):
            self.motions = self.motions['arr_0']
        self.motions = self.motions[None][0]
        self.rot2xyz = Rotation2xyz(device='cpu')
        self.faces = self.rot2xyz.smpl_model.faces
        
        # Handle different motion array shapes
        motion_array = self.motions['motion']
        if len(motion_array.shape) == 5:  # If shape is [num_rep, num_samples, njoints, nfeats, nframes]
            motion_array = motion_array[rep_idx]  # Remove repetition dimension
        
        self.bs, self.njoints, self.nfeats, self.nframes = motion_array.shape
        self.opt_cache = {}
        self.sample_idx = sample_idx
        self.total_num_samples = self.motions['num_samples']
        self.rep_idx = rep_idx
        self.absl_idx = self.sample_idx  # Modified to use sample_idx directly since we already handled rep_idx
        self.num_frames = motion_array[self.absl_idx].shape[-1]
        self.j2s = joints2smpl(num_frames=self.num_frames, device_id=device, cuda=cuda)

        if self.nfeats == 3:
            print(f'Running SMPLify For sample [{sample_idx}], repetition [{rep_idx}], it may take a few minutes.')
            motion_tensor, opt_dict = self.j2s.joint2smpl(motion_array[self.absl_idx].transpose(2, 0, 1))  # [nframes, njoints, 3]
            motion_array = motion_tensor.cpu().numpy()
        elif self.nfeats == 6:
            motion_array = motion_array[[self.absl_idx]]
            
        self.motions['motion'] = motion_array
        self.bs, self.njoints, self.nfeats, self.nframes = motion_array.shape
        
        # Ensure real_num_frames is an integer
        if isinstance(self.motions['lengths'], np.ndarray):
            self.real_num_frames = int(self.motions['lengths'][self.absl_idx])
        else:
            self.real_num_frames = int(self.motions['lengths'])

        self.vertices = self.rot2xyz(torch.tensor(self.motions['motion']), mask=None,
                                     pose_rep='rot6d', translation=True, glob=True,
                                     jointstype='vertices',
                                     # jointstype='smpl',  # for joint locations
                                     vertstrans=True)
        self.root_loc = self.motions['motion'][:, -1, :3, :].reshape(1, 1, 3, -1)

        # import pdb; pdb.set_trace()
        # self.vertices += self.root_loc
        # self.vertices[:, :, 1, :] += self.root_loc[:, :, 1, :]

    def get_vertices(self, sample_i, frame_i):
        return self.vertices[sample_i, :, :, frame_i].squeeze().tolist()

    def get_trimesh(self, sample_i, frame_i):
        return Trimesh(vertices=self.get_vertices(sample_i, frame_i),
                       faces=self.faces)
    
    def get_traj_sphere(self, mesh):
        # import pdb; pdb.set_trace()
        root_posi = np.copy(mesh.vertices).mean(0) # (6000, 3)
        # import pdb; pdb.set_trace()
        # root_posi[1] = mesh.vertices.min(0)[1] + 0.1
        root_posi[1]  = self.vertices.numpy().min(axis=(0, 1, 3))[1] + 0.1
        mesh = trimesh.primitives.Sphere(radius=0.05, center=root_posi, transform=None, subdivisions=1)
        return mesh

    def save_obj(self, save_path, frame_i):
        mesh = self.get_trimesh(0, frame_i)
        ground_sph_mesh = self.get_traj_sphere(mesh)
        loc_obj_name = os.path.splitext(os.path.basename(save_path))[0] + "_ground_loc.obj"
        ground_save_path = os.path.join(os.path.dirname(save_path), "loc", loc_obj_name)
        with open(save_path, 'w') as fw:
            mesh.export(fw, 'obj')
        with open(ground_save_path, 'w') as fw:
            ground_sph_mesh.export(fw, 'obj')
        return save_path
    
    def save_npy(self, save_path):
        # Get the motion data and ensure correct shape
        motion_data = self.motions['motion']
        if len(motion_data.shape) == 5:  # [num_rep, num_samples, njoints, nfeats, nframes]
            motion_data = motion_data[self.rep_idx, self.sample_idx]
        else:  # [num_samples, njoints, nfeats, nframes]
            motion_data = motion_data[self.sample_idx]

        data_dict = {
            'motion': motion_data[:, :, :self.real_num_frames],
            'thetas': motion_data[:-1, :, :self.real_num_frames],
            'root_translation': motion_data[-1, :3, :self.real_num_frames],
            'faces': self.faces,
            'vertices': self.vertices[0, :, :, :self.real_num_frames],
            'text': self.motions['text'][self.rep_idx] if len(self.motions['text'].shape) > 1 else self.motions['text'][0],
            'length': self.real_num_frames,
        }
        np.save(save_path, data_dict)
