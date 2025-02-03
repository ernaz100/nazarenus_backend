import json
import numpy as np
import torch
from smplx import SMPL
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import os

class PoseEstimator:
    def __init__(self, model_path="./models/smpl"):
        self.smpl_model = SMPL(model_path, gender='female')
        self.joint_mapping = {
            0: 0,   # pelvis 
            1: 1,   # left hip 
            2: 4,   # left knee
            3: 7,   # left ankle
            4: 10,  # left foot
            5: 2,   # right hip
            6: 5,   # right knee
            7: 8,   # right ankle
            8: 11,  # right foot
            9: 3,  # spine1
            10: 6,  # spine2
            11: 9,  # spine3
            12: 12, # neck
            13: 15, # head
            14: 13, # left collar
            15: 16, # left shoulder
            16: 18, # left elbow
            17: 20, # left wrist
            18: 22, # left hand
            19: 14, # right collar
            20: 17, # right shoulder
            21: 19, # right elbow
            22: 21, # right wrist
            23: 23, # right hand
        }
        
        # Get the kinematic tree from the SMPL model
        self.parents = self.smpl_model.parents.numpy()

    def forward_kinematics(self, pose_params):
        """ Compute joint positions from given pose parameters """
        # Separate global orientation (root) and body pose
        global_orient = pose_params[:3].reshape(1, 3)
        body_pose = pose_params[3:].reshape(1, 69)
        
        # Convert to torch tensors
        global_orient = torch.tensor(global_orient, dtype=torch.float32)
        body_pose = torch.tensor(body_pose, dtype=torch.float32)
        
        output = self.smpl_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=torch.zeros(1, 10),
            return_verts=False
        )
        # takes the first 24 joints from the SMPL model as in the forward pass hands, face and feet joints are concatenated 
        return output.joints[0].detach().cpu().numpy()[:24]


    def loss_function(self, pose_params, target_joints, selected_joint=None):
        """Compute the loss between current and target joint positions"""
        predicted_joints = self.forward_kinematics(pose_params)
        
        if selected_joint is not None:
            # Get joints in the kinematic chain
            chain = self.get_kinematic_chain(selected_joint)
            loss = 0
            for idx in range(len(predicted_joints)):
                weight = 1.0
                if idx == self.joint_mapping[selected_joint]:
                    weight = 100.0  # Highest weight for selected joint
                elif self.joint_mapping[idx] in chain:
                    weight = 10.0   # Medium weight for joints in chain
                else:
                    weight = 0.1    # Low weight for other joints to keep them stable
                
                loss += weight * np.linalg.norm(predicted_joints[idx] - target_joints[idx])
            return loss
    
    def estimate_pose(self, target_joints, selected_joint=None):
        """ Optimize pose to match given joint positions """
        # Remap joints to SMPL ordering
        remapped_joints = self.remap_joints(target_joints)
        
        # Initialize with current pose if moving from previous position
        initial_pose = np.zeros(72)
        
        # Add tighter bounds for better stability
        bounds = []
        # Global orientation can be full rotation
        bounds.extend([(-np.pi, np.pi)] * 3)
        # Joint rotations should be more limited
        bounds.extend([(-np.pi/4, np.pi/4)] * 69)
        
        # Try multiple optimizations with different initial poses
        best_result = None
        best_loss = float('inf')
        
        for _ in range(3):  # Try 3 different initializations
            # Add some random noise to initialization
            current_initial = initial_pose + np.random.normal(0, 0.1, 72)
            
            result = minimize(
                self.loss_function, 
                current_initial, 
                args=(remapped_joints, selected_joint), 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100}
            )
            
            if result.fun < best_loss:
                best_loss = result.fun
                best_result = result
        
        global_orient = torch.tensor(best_result.x[:3].reshape(1, 3), dtype=torch.float32)
        body_pose = torch.tensor(best_result.x[3:].reshape(1, 69), dtype=torch.float32)
        
        # Get model output with vertices and joints
        with torch.no_grad():
            best_joint_pos = self.smpl_model(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=torch.zeros((1, 10), dtype=torch.float32),
                return_verts=False,
                return_full_pose=False
            )
        # takes the first 24 joints from the SMPL model as in the forward pass hands, face and feet joints are concatenated 
        joints = best_joint_pos.joints[0].detach().cpu().numpy()[:24]
        
        # Make positions relative to pelvis
        pelvis_position = joints[0]
        relative_joints = joints - pelvis_position[None, :]
        
        # Remap joints to match frontend ordering
        frontend_joints = self.remap_joints_to_frontend(relative_joints)
        
        # Remap pose parameters back to frontend ordering
        frontend_pose_params = self.remap_pose_params_back(best_result.x)
        
        return frontend_joints, frontend_pose_params


    def visualize_pose(self, pose_params=None, selected_joint=None, title="pose_visualization.png"):
        """Visualize the SMPL model pose using Matplotlib and save as image"""
        if pose_params is None:
            pose_params = np.zeros(72)  # Default T-pose
            
        # Get joints and vertices
        global_orient = torch.tensor(pose_params[:3].reshape(1, 3), dtype=torch.float32)
        body_pose = torch.tensor(pose_params[3:].reshape(1, 69), dtype=torch.float32)
        
        output = self.smpl_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=torch.zeros(1, 10),
            return_verts=True
        )
        
        vertices = output.vertices[0].detach().cpu().numpy()
        joints = output.joints[0].detach().cpu().numpy()[:24]
        
        # Create color array for joints (all red by default)
        colors = ['r'] * len(joints)
        sizes = [50] * len(joints)  # Default size for joints
        if selected_joint is not None:
            # Map frontend index to SMPL index and set color to green
            smpl_joint_idx = self.joint_mapping[selected_joint]
            colors[smpl_joint_idx] = 'lime'  # Brighter green
            sizes[smpl_joint_idx] = 200  # Much larger size
        
        # Create figure with multiple views
        fig = plt.figure(figsize=(15, 5))
        
        # Front view (looking at Y-Z plane)
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='b', marker='.', s=1)
        for joint, color, size in zip(joints, colors, sizes):
            ax1.scatter(joint[0], joint[1], joint[2], c=color, marker='o', s=size)
        ax1.view_init(elev=90, azim=-90)  # Looking at Y-Z plane
        ax1.set_title('Front View')
        
        # Side view (looking at X-Z plane)
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(vertices[:, 0], vertices[:, 2], vertices[:, 1], c='b', marker='.', s=1)
        for joint, color, size in zip(joints, colors, sizes):
            ax2.scatter(joint[0], joint[2], joint[1], c=color, marker='o', s=size)
        ax2.view_init(elev=0, azim=0)  # Looking at X-Z plane
        ax2.set_title('Side View')
        
        # Top view (looking down at X-Y plane)
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='b', marker='.', s=1)
        for joint, color, size in zip(joints, colors, sizes):
            ax3.scatter(joint[0], joint[1], joint[2], c=color, marker='o', s=size)
        ax3.view_init(elev=0, azim=-90)  
        ax3.set_title('Top View')
        
        # Set consistent axes limits
        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_box_aspect([1,1,1])
        
        # Save the figure
        plt.tight_layout()
        save_path = title
        plt.savefig(save_path)
        plt.close()
        
        return save_path

    def visualize_joints(self, joints, selected_joint=None, title="input_joints.png"):
        """Visualize input joint positions"""
        # Create figure with multiple views
        fig = plt.figure(figsize=(15, 5))

        # Create color array (all red by default)
        colors = ['r'] * len(joints)
        if selected_joint is not None:
            # Set selected joint color to green
            colors[selected_joint] = 'g'
        
        # Front view 
        ax1 = fig.add_subplot(131, projection='3d')
        for i, (joint, color) in enumerate(zip(joints, colors)):
            ax1.scatter(joint[0], joint[1], joint[2], c=color, marker='o')
        ax1.view_init(elev=90, azim=-90)  
        ax1.set_title('Front View')
        
        # Side view
        ax2 = fig.add_subplot(132, projection='3d')
        for i, (joint, color) in enumerate(zip(joints, colors)):
            ax2.scatter(joint[0], joint[2], joint[1], c=color, marker='o')
        ax2.view_init(elev=0, azim=0) 
        ax2.set_title('Side View')
        
        # Top view 
        ax3 = fig.add_subplot(133, projection='3d')
        for i, (joint, color) in enumerate(zip(joints, colors)):
            ax3.scatter(joint[0], joint[1], joint[2], c=color, marker='o')
        ax3.view_init(elev=0, azim=-90)  
        ax3.set_title('Top View')
        
        # Set consistent axes limits and labels
        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_box_aspect([1,1,1])
        
        # Save the figure
        plt.tight_layout()
        save_path = title
        plt.savefig(save_path)
        plt.close()
        
        return save_path

    def export_pose_glb(self, pose_params=None, output_path="pose.glb"):
        """Export the SMPL model pose as a GLB file with joint positions"""
        if pose_params is None:
            pose_params = np.zeros(72)  # Default T-pose
            
        # Convert pose parameters to torch tensors with correct shape and type
        global_orient = torch.tensor(pose_params[:3].reshape(1, 3), dtype=torch.float32)
        body_pose = torch.tensor(pose_params[3:].reshape(1, 69), dtype=torch.float32)
        
        # Get model output with vertices and joints
        with torch.no_grad():
            output = self.smpl_model(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=torch.zeros((1, 10), dtype=torch.float32),
                return_verts=True,
                return_full_pose=True
            )
        
        # Get vertices, faces, and joints
        vertices = output.vertices[0].detach().cpu().numpy()
        faces = self.smpl_model.faces
        joints = output.joints[0].detach().cpu().numpy()
        
        # Create mesh for the body
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Create small spheres for joints
        joint_meshes = []
        for joint_pos in joints:
            joint_sphere = trimesh.primitives.Sphere(radius=0.02, center=joint_pos)
            joint_meshes.append(joint_sphere)
        
        # Combine body mesh with joint spheres
        combined_mesh = trimesh.util.concatenate([mesh] + joint_meshes)
        
        # Rotate mesh 90 degrees around X-axis to match standard orientation
        rotation = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
        combined_mesh.apply_transform(rotation)
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        combined_mesh.export(output_path)
        return output_path

    def remap_joints(self, input_joints):
        """
        Remap joints from frontend ordering to SMPL ordering
        
        Args:
            input_joints: numpy array of shape (24, 3) in frontend order
        
        Returns:
            remapped_joints: numpy array of shape (24, 3) in SMPL order
        """
        if input_joints.shape != (24, 3):
            raise ValueError(f"Expected input_joints shape (24, 3), got {input_joints.shape}")

        remapped_joints = np.zeros((24, 3))
        for frontend_idx, smpl_idx in self.joint_mapping.items():
            remapped_joints[smpl_idx] = input_joints[frontend_idx]
        return remapped_joints

    def remap_pose_params_back(self, smpl_pose_params):
        """
        Remap pose parameters from SMPL ordering back to frontend ordering
        
        Args:
            smpl_pose_params: numpy array of shape (72,) in SMPL order
            (first 3 values are global orientation, then 23 joints * 3 rotation params)
        
        Returns:
            frontend_pose_params: numpy array of shape (72,) in frontend order
        """
        if smpl_pose_params.shape != (72,):
            raise ValueError(f"Expected smpl_pose_params shape (72,), got {smpl_pose_params.shape}")

        # Create output array
        frontend_pose_params = np.zeros(72)
        
        # Keep global orientation (first 3 values) unchanged
        frontend_pose_params[:3] = smpl_pose_params[:3]
        
        # Remap the joint rotations (remaining 69 values, 3 per joint)
        for frontend_idx, smpl_idx in self.joint_mapping.items():
            src_idx = smpl_idx * 3 + 3  # +3 to skip global orientation
            dst_idx = frontend_idx * 3 + 3
            frontend_pose_params[dst_idx:dst_idx+3] = smpl_pose_params[src_idx:src_idx+3]
        
        return frontend_pose_params

    def remap_joints_to_frontend(self, joints):
        """
        Remap joints from SMPL ordering back to frontend ordering
        
        Args:
            joints: numpy array of shape (24, 3) in SMPL order
        
        Returns:
            frontend_joints: numpy array of shape (24, 3) in frontend order
        """
        if joints.shape != (24, 3):
            raise ValueError(f"Expected joints shape (24, 3), got {joints.shape}")

        frontend_joints = np.zeros_like(joints)
        # Reverse the mapping: for each frontend_idx, smpl_idx pair, 
        # put the SMPL joint at frontend position
        for frontend_idx, smpl_idx in self.joint_mapping.items():
            frontend_joints[frontend_idx] = joints[smpl_idx]
        return frontend_joints

    def get_kinematic_chain(self, joint_idx):
        """Get all joints in the kinematic chain of the selected joint using SMPL's parent array"""
        chain = set()
        
        # Map frontend index to SMPL index
        smpl_idx = self.joint_mapping[joint_idx]
        
        # Get ancestors (going up the tree)
        current = smpl_idx
        while current != -1:  # -1 is the root in SMPL
            chain.add(current)
            current = self.parents[current]
        
        # Get descendants (going down the tree)
        def add_children(idx):
            children = np.where(self.parents == idx)[0]
            for child in children:
                chain.add(child)
                add_children(child)
        
        add_children(smpl_idx)
        return chain
