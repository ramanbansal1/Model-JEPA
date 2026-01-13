import torch
import trimesh
import numpy as np
import os
import sys
import subprocess
import argparse
import logging
import glob
from torch.utils.data.sampler import Sampler
import MinkowskiEngine as ME
from utils import extract_points


class ModelNet40Dataset(torch.utils.data.Dataset):
    def __init__(self, phase, transform=None, config=None):
        self.phase = phase
        self.files = []
        self.cache = {}
        self.data_objects = []
        self.transform = transform
        self.resolution = config.resolution
        self.last_cache_percent = 0

        self.root = "./ModelNet40"
        fnames = glob.glob(os.path.join(self.root, "airplane/train/*.off"))
        fnames = sorted([os.path.relpath(fname, self.root) for fname in fnames])
        self.files = fnames
        assert len(self.files) > 0, "No file loaded"
        print(
            f"Loading the subset {phase} from {self.root} with {len(self.files)} files"
        )
        self.density = 30000

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mesh_file = os.path.join(self.root, self.files[idx])
        
        if idx in self.cache:
            xyz = self.cache[idx]
        else:
            # Load a mesh, over sample, copy, rotate, voxelization
            assert os.path.exists(mesh_file)
            xyz = extract_points(mesh_file, self.density)
            
            self.cache[idx] = xyz

        # Use color or other features if available
        feats = np.ones((len(xyz), 1))

        if len(xyz) < 1000:
            print(
                f"Skipping {mesh_file}: does not have sufficient CAD sampling density after resampling: {len(xyz)}."
            )
            return None

        if self.transform:
            xyz, feats = self.transform(xyz, feats)

        # Get coords
        xyz = xyz * self.resolution
        coords, inds = ME.utils.sparse_quantize(xyz, return_index=True)

        return (coords, xyz[inds], idx)
    


class CollationAndTransformation:
    def __init__(self, resolution):
        self.resolution = resolution

    def random_crop(self, coords_list):
        crop_coords_list = []
        for coords in coords_list:
            sel = coords[:, 0] < self.resolution / 3
            crop_coords_list.append(coords[sel])
        return crop_coords_list

    def __call__(self, list_data):
        coords, feats, labels = list(zip(*list_data))
        coords = self.random_crop(coords)

        # Concatenate all lists
        return {
            "coords": ME.utils.batched_coordinates(coords),
            "xyzs": [torch.from_numpy(feat).float() for feat in feats],
            "cropped_coords": coords,
            "labels": torch.LongTensor(labels),
        }
    

def make_data_loader(
    phase, batch_size, shuffle, num_workers, repeat, config
):
    dset = ModelNet40Dataset(phase, config=config)

    args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": CollationAndTransformation(config.resolution),
        "pin_memory": False,
        "drop_last": False,
    }

    if repeat:
        args["sampler"] = InfSampler(dset, shuffle)
    else:
        args["shuffle"] = shuffle

    loader = torch.utils.data.DataLoader(dset, **args)

    return loader


class InfSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)