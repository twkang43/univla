import numpy as np
import torch
import os
import glob
import h5py
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
import fnmatch
import subprocess
import pickle
import re
from datetime import datetime
import cv2
import logging
from PIL import Image
from einops import rearrange, repeat
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision
import random
from dataclasses import dataclass
from typing import Sequence, Dict


logger = logging.getLogger(__name__)
# Example
language_tasks = [
    'Put the screwdriver in the cabinet and close the cabinet',
]

class HDF5Dataset(torch.utils.data.Dataset):
    
    def __init__(self, episode_ids, 
                dataset_dir, 
                camera_names, 
                norm_stats, 
                window_size = 16,
                min_window_size = 16,
                max_window_size = 16,
                n_demos_per_data=None,
                image_transform = None,
                other_config=()) -> None:
        
        super(HDF5Dataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.other_config = other_config
        self.chunk_size = window_size
        self.window_size = window_size
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.resize_img = torchvision.transforms.Resize((224, 224))
        self.image_transform_lam = torchvision.transforms.ToTensor()
        self.image_transform = image_transform
        self.episode_lens = []
        self.image_dict, self.qpos, self.action, self.tasks_embedding = self.load_all_episodes(dataset_dir, n_demos_per_data)
        self.color_aug = torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)


    def __len__(self):
        return len(self.action)

    def load_all_episodes(self, dataset_paths, n_demos_per_data=None):
        image_dict = dict()
        for cam_name in self.camera_names:
            image_dict[cam_name] = []
        qpos = []
        actions = []
        instructions = []

        for dataset_path in dataset_paths:
            print(f"processing {dataset_path}")
            try:
                with h5py.File(dataset_path, 'r') as root:
                    # Handle different HDF5 structures by finding the main data group
                    if 'data' in root:
                        data_group = root['data']
                    else:
                        # If no 'data' group, assume root is the main group
                        data_group = root

                    # Some files have demos, some are single episodes.
                    # If no 'demo_xx' keys, treat the file as a single episode.
                    demo_keys = [k for k in data_group.keys() if k.startswith('demo_')]
                    
                    if not demo_keys:
                        demo_keys = [None] # Use None as a placeholder key for single-episode files
                    elif (n_demos_per_data is not None) and (0 < n_demos_per_data): # TODO: revise data truncation logic
                        demo_keys = demo_keys[:n_demos_per_data]

                    for demo_key in tqdm(demo_keys, desc=f"Loading {dataset_path}", leave=False):
                        episode_group = data_group[demo_key] if demo_key else data_group

                        # Find the actual data source, handling weirdly nested 'data' groups
                        if 'obs' in episode_group and 'actions' in episode_group:
                            data_source = episode_group
                        elif 'data' in episode_group and 'obs' in episode_group['data'] and 'actions' in episode_group['data']:
                            data_source = episode_group['data']
                        else:
                            print(f"Warning: Skipping episode in {dataset_path} (key: {demo_key}) due to missing 'obs' or 'actions'. Keys: {list(episode_group.keys())}")
                            continue
                        
                        try:
                            action_data = data_source['actions'][()]
                            qpos_data = data_source['obs/joint_states'][()]
                            
                            current_episode_len = action_data.shape[0]
                            self.episode_lens.append(current_episode_len)

                            qpos.append(torch.from_numpy(qpos_data))
                            actions.append(torch.from_numpy(action_data))

                            # We store file names as task instructions, please adjust accordingly
                            file_name = dataset_path.split('/')[-1]
                            task_instruction = file_name.split('+')[0].replace('_', ' ')
                            instructions.append(task_instruction)
                            
                            compressed = 'compress' in data_source.attrs

                            for cam_name in self.camera_names:
                                image_one_cam = []
                                image_hdf5_data = data_source[f'obs/{cam_name}']
                                for i_img in range(image_hdf5_data.shape[0]):
                                    if compressed:
                                        raw_image = cv2.imdecode(image_hdf5_data[i_img], 1)
                                    else:
                                        raw_image = image_hdf5_data[i_img]
                                    flipped_image = torch.flip(torch.from_numpy(raw_image), dims=(-1,))
                                    resized_image = F.interpolate(flipped_image.permute(2, 0, 1).unsqueeze(0).float(), size=(224, 224), mode='bilinear', align_corners=False)
                                    image_one_cam.append(resized_image[0])
                                image_dict[cam_name].append(torch.stack(image_one_cam, dim=0))

                        except KeyError as e:
                            print(f"Warning: KeyError when processing {dataset_path} (key: {demo_key}): {e}. Skipping.")
                            if self.episode_lens: self.episode_lens.pop() # Rollback
                            continue
            
            except Exception as e:
                print(f"Error loading {dataset_path}: {e}")

        return image_dict, qpos, actions, instructions


    def __getitem__(self, clip_index):

        extra_frame_num = random.randint(0, 1)
        window_size = self.window_size + extra_frame_num
        
        episode_len = self.episode_lens[clip_index]
        if episode_len <= window_size:
            # If the episode is too short, use the whole episode
            image_index = 0
            window_size = episode_len
        else:
            image_index = np.random.choice(episode_len - window_size)

        actions_chunking = torch.zeros((self.chunk_size, self.action[clip_index].shape[-1]))
        is_not_padding = torch.zeros((self.chunk_size,))
        
        # Use the correct episode length for slicing
        action_slice = self.action[clip_index][image_index : image_index + self.chunk_size]
        actions_chunking[:len(action_slice)] = action_slice
        qpos_chunking = self.qpos[clip_index][image_index]

        cam_name = "agentview_rgb"
        image_chunking = self.image_dict[cam_name][clip_index][image_index : image_index + window_size]
        image_vla = Image.fromarray(np.transpose(image_chunking[extra_frame_num].cpu().numpy().astype(np.uint8), (1, 2, 0)))
        image_vla = self.color_aug(image_vla)
        goal_image = Image.fromarray(np.transpose(image_chunking[-1].cpu().numpy().astype(np.uint8), (1, 2, 0)))
        pixel_values = self.image_transform(image_vla)
        
        initial_pixel_values = self.image_transform_lam(self.resize_img(image_vla))
        target_pixel_values = self.image_transform_lam(self.resize_img(goal_image))
        
        initial_pixel_values_hist, target_pixel_values_hist = None, None
        if extra_frame_num > 0:
            hist_frame_prev = Image.fromarray(np.transpose(image_chunking[0].cpu().numpy().astype(np.uint8), (1, 2, 0)))
            hist_frame_goal = Image.fromarray(np.transpose(image_chunking[self.min_window_size].cpu().numpy().astype(np.uint8), (1, 2, 0)))
            initial_pixel_values_hist = self.image_transform_lam(self.resize_img(hist_frame_prev))
            target_pixel_values_hist = self.image_transform_lam(self.resize_img(hist_frame_goal))
        
        is_not_padding[:len(action_slice)] = 1
        
        # normalize actions and change dtype to float
        qpos_tensor = qpos_chunking.float()
        action_tensor = actions_chunking.float()
        action_tensor = (action_tensor - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_tensor = (qpos_tensor - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        task_embed = self.tasks_embedding[clip_index]
        
        dataset_name = 'agilex'
        
        return dict(pixel_values=pixel_values, initial_pixel_values=initial_pixel_values, target_pixel_values=target_pixel_values, 
                    initial_pixel_values_hist=initial_pixel_values_hist, target_pixel_values_hist=target_pixel_values_hist,
                    dataset_name=dataset_name, actions=action_tensor, lang=task_embed, proprio=qpos_tensor)


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        
        initial_pixel_values = [instance["initial_pixel_values"] for instance in instances]
        target_pixel_values = [instance["target_pixel_values"] for instance in instances]

        initial_pixel_values_hist, target_pixel_values_hist = [], []
        with_hist = []
        for instance in instances:
            if instance["initial_pixel_values_hist"] is not None:
                initial_pixel_values_hist.append(instance["initial_pixel_values_hist"])
                target_pixel_values_hist.append(instance["target_pixel_values_hist"])
                with_hist.append(torch.tensor(True))
            else:
                with_hist.append(torch.tensor(False))     



        pixel_values = [instance["pixel_values"] for instance in instances]
        if "dataset_name" in instances[0]:
            dataset_names = [instance["dataset_name"] for instance in instances]
        else:
            dataset_names = None


        # For low-level policy training
        actions = [instance["actions"] for instance in instances]
        actions = torch.stack(actions, dim=0)

        proprio = [instance["proprio"] for instance in instances]
        proprio = torch.stack(proprio, dim=0)

        instructions = [instance["lang"] for instance in instances]


        # [Contract] For VLA Training =>> No "Unimodal" Data!
        assert all([pv is not None for pv in pixel_values]), "Invalid VLA Example with `pixel_values = None`!"

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        pixel_values = torch.stack(pixel_values)
        initial_pixel_values = torch.stack(initial_pixel_values)
        target_pixel_values = torch.stack(target_pixel_values)
        initial_pixel_values_hist = torch.stack(initial_pixel_values_hist) if len(initial_pixel_values_hist) > 0 else []
        target_pixel_values_hist = torch.stack(target_pixel_values_hist) if len(target_pixel_values_hist) > 0 else []
        with_hist = torch.stack(with_hist)

        output = dict(
            pixel_values=pixel_values,
            initial_pixel_values=initial_pixel_values,
            target_pixel_values=target_pixel_values,
            initial_pixel_values_hist=initial_pixel_values_hist,
            target_pixel_values_hist=target_pixel_values_hist,
            instructions=instructions,
            with_hist=with_hist,
            actions=actions,
            proprio=proprio
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names
        return output


def load_data_univla(dataset_paths, camera_names, batch_size_train, action_tokenizer, processor, window_size,     
        min_window_size, max_window_size, n_demos_per_data, image_transform, other_info=()):

    num_episodes = len(dataset_paths)
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_paths, other_info)

    train_dataset = HDF5Dataset(train_indices, dataset_paths, camera_names, norm_stats,
        window_size = window_size,
        min_window_size = min_window_size,
        max_window_size = max_window_size,
        n_demos_per_data = n_demos_per_data,
        image_transform = image_transform,
    )

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size_train, 
        shuffle=True, 
        pin_memory=False, 
        num_workers=8, 
        prefetch_factor=2, 
        collate_fn=collator
    )


    return train_dataloader, norm_stats


def find_all_hdf5(dataset_dir, n_data, skip_mirrored_data=True):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename: continue
            if skip_mirrored_data and 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files[:n_data]

def get_norm_stats(dataset_paths, other_config=()):
    all_qpos_data = []
    all_action_data = []
    for dataset_path in dataset_paths:
        try:
            with h5py.File(dataset_path, 'r') as root:
                # Handle different HDF5 structures
                if 'data' in root:
                    data_group = root['data']
                else:
                    data_group = root

                demo_keys = [k for k in data_group.keys() if k.startswith('demo_')]
                if not demo_keys:
                    demo_keys = [None]

                for demo_key in demo_keys:
                    episode_group = data_group[demo_key] if demo_key else data_group
                    
                    if 'obs' in episode_group and 'actions' in episode_group:
                        data_source = episode_group
                    elif 'data' in episode_group and 'obs' in episode_group['data'] and 'actions' in episode_group['data']:
                        data_source = episode_group['data']
                    else:
                        continue

                    qpos = data_source['obs/joint_states'][()]
                    action = data_source['actions'][()]
                    
                    all_qpos_data.append(torch.from_numpy(qpos))
                    all_action_data.append(torch.from_numpy(action))
                    
        except Exception as e:
            print(f"Error processing file {dataset_path} for norm_stats: {e}")

    if not all_action_data:
        # Return dummy stats if no data was loaded
        print("Warning: No data loaded for normalization. Returning dummy stats.")
        dummy_action = np.zeros(7)
        dummy_qpos = np.zeros(7)
        return {
            "action_mean": dummy_action, "action_std": np.ones_like(dummy_action),
            "qpos_mean": dummy_qpos, "qpos_std": np.ones_like(dummy_qpos),
            "example_qpos": dummy_qpos, "action_max": np.ones_like(dummy_action), "action_min": -np.ones_like(dummy_action)
        }

    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # normalize action data
    action_mean = all_action_data.mean(dim=0, keepdim=True)
    action_std = all_action_data.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std = all_qpos_data.std(dim=0, keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping
    
    # Min-max norm action datra
    action_max = all_action_data.max(dim=0, keepdim=True)[0]
    action_min = all_action_data.min(dim=0, keepdim=True)[0]

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos, "action_max":action_max.numpy().squeeze(), "action_min":action_min.numpy().squeeze()}

    return stats

def get_key_info(path):
    if '.pkl' not in path:
        path = os.path.join(path, f'key_info.pkl')
    with open(path, 'rb') as f:
        key_info = pickle.load(f)
    return key_info

def get_init_states(path_first_episode):
    if os.path.exists(path_first_episode):
        with h5py.File(path_first_episode, 'r') as root:
            qpos = root['/observations/qpos'][0]
            action = root['/action'][0]
    else:
        # dir is info dir
        key_info_path = os.path.join(dir, f'key_info.pkl')
        with open(key_info_path, 'rb') as f:
            key_info = pickle.load(f)
            qpos = key_info['init_info']['init_joint']
            action = key_info['init_info']['init_action']
    return qpos, action


