"""Custom Dataset and DataLoader classes.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, default_collate
import jax.numpy as jnp
from jax.tree_util import tree_map

############################
##  Custom Dataset Class  ##
############################

class LandscapeSimulationDataset(Dataset):
    """A collection of data generated via landscape simulations.
    
    Each simulation consists of generating a number of paths between time
    t0=0 and t1, sampling the state of each path at a set of time intervals t_i,
    and thus capturing a pair (t_i, X_i), where X_i is an N by d state matrix.
    The data then consists of tuples (t_{i}, X_{i}, t_{i+1}, X_{i+1}), which 
    represent the evolution in state between consecutive sampling times.

    """
    
    def __init__(self, datdir, nsims, dim, 
                 transform=None, target_transform=None, **kwargs):
        #~~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~#
        simprefix = kwargs.get('simprefix', 'sim')
        dtype = kwargs.get('dtype', torch.float32)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.nsims = nsims
        self.dim = dim
        self.transform = transform
        self.target_transform = target_transform
        self.dtype = dtype
        self._load_data(datdir, nsims, simprefix=simprefix)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        t0, x0, t1, x1, ps = data
        # Transform input x
        if self.transform == 'tensor':
            x = np.concatenate([[t0], [t1], x0.flatten(), ps])
            # x = torch.tensor(x, dtype=self.dtype, requires_grad=True)
            x = torch.tensor(x, dtype=self.dtype)
        elif self.transform:
            x = self.transform(*data)
        else:
            x = t0, x0, t1, ps
        # Transform target y, the final distribution
        if self.target_transform == 'tensor':
            y = torch.tensor(x1, dtype=self.dtype)
        elif self.target_transform:
            y = self.target_transform(*data)
        else:
            y = x1
        return x, y
    
    def preview(self, idx, **kwargs):
        """Plot a data item."""
        #~~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~#
        ax = kwargs.get('ax', None)
        col1 = kwargs.get('col1', 'b')
        col2 = kwargs.get('col2', 'r')
        size = kwargs.get('size', 2)
        xlims = kwargs.get('xlims', None)
        ylims = kwargs.get('ylims', None)
        show = kwargs.get('show', True)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        data = self.dataset[idx]
        t0, x0, t1, x1, ps = data
        if ax is None: fig, ax = plt.subplots(1, 1)
        ax.plot(x0[:,0], x0[:,1], '.', 
                c=col1, markersize=size, label=f"$t={t0:.3g}$")
        ax.plot(x1[:,0], x1[:,1], '.', 
                c=col2, markersize=size, label=f"$t={t1:.3g}$")
        if xlims is not None: ax.set_xlim(*xlims)
        if ylims is not None: ax.set_ylim(*ylims)
        ax.set_xlabel(f"$x$")
        ax.set_ylabel(f"$y$")
        ax.set_title(f"datapoint {idx}/{len(self)}")
        s = f"$t:{t0:.4g}\\to{t1:.4g}$\
            \n$t^*={ps[0]:.3g}$\
            \n$p_0=[{ps[1]:.3g}, {ps[2]:.3g}]$\
            \n$p_1=[{ps[3]:.3g}, {ps[4]:.3g}]$"
        ax.text(0.02, 0.02, s, fontsize=8, transform=ax.transAxes)
        ax.legend()
        if show: plt.show()
        return ax

    def animate(self, simidx, interval=50, **kwargs):
        """Animate a given simulation"""
        idx0 = int(simidx * len(self) // self.nsims)
        idx1 = idx0 + int(len(self) // self.nsims)
        video = []
        for idx in range(idx0, idx1):
            ax = self.preview(idx, **kwargs)
            ax.figure.canvas.draw()
            data = np.frombuffer(ax.figure.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(ax.figure.canvas.get_width_height()[::-1] + (3,))
            video.append(data)
            plt.close()
        video = np.array(video)

        fig = plt.figure()
        plt.axis('off')
        plt.tight_layout()
        im = plt.imshow(video[0,:,:,:])
        plt.close() 
        def init():
            im.set_data(video[0,:,:,:])

        def ani_func(i):
            im.set_data(video[i,:,:,:])
            return im

        anim = animation.FuncAnimation(
            fig, ani_func, init_func=init, 
            frames=video.shape[0],
            interval=interval,
        )
        return anim.to_html5_video()

    ######################
    ##  Helper Methods  ##
    ######################
    
    def _load_data(self, datdir, nsims, simprefix='sim'):
        ts_all = []
        xs_all = []
        ps_all = []
        dataset = []
        for i in range(nsims):
            dirpath = f"{datdir}/{simprefix}{i}"
            assert os.path.isdir(dirpath), f"Not a directory: {dirpath}"
            ts = np.load(f"{dirpath}/ts.npy")
            xs = np.load(f"{dirpath}/xs.npy")
            ps = np.load(f"{dirpath}/ps.npy")
            p_params = np.load(f"{dirpath}/p_params.npy")
            ts_all.append(ts)
            xs_all.append(xs)
            ps_all.append(ps)
            self._add_sim_data_to_dataset(dataset, ts=ts, xs=xs, ps=ps, 
                                          p_params=p_params)
        self.dataset = np.array(dataset, dtype=object)
        self.ts_all = ts_all
        self.xs_all = xs_all
        self.ps_all = ps_all

    def _add_sim_data_to_dataset(self, dataset, ts, xs, ps, p_params):
        ntimes = len(ts)
        for i in range(ntimes - 1):
            x0, x1 = xs[i], xs[i + 1]
            t0, t1 = ts[i], ts[i + 1]
            dataset.append((t0, x0, t1, x1, p_params))


###############################
##  Custom DataLoader Class  ##
###############################

class NumpyLoader(DataLoader):
    """Custom DataLoader to return numpy arrays instead of torch tensors.
    """

    def __init__(self, dataset, batch_size=1,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=lambda b: tree_map(jnp.asarray, default_collate(b)),
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn
        )

def get_dataloaders(
        datdir_train,
        datdir_valid,
        nsims_train,
        nsims_valid,
        batch_size_train=1,
        batch_size_valid=1,
        ndims=2,
        dtype=np.float64,
        shuffle_train=True,
        shuffle_valid=False,
        return_datasets=False,
    ):
    """TODO

    Args:
        datdir_train (_type_): _description_
        datdir_valid (_type_): _description_
        nsims_train (_type_): _description_
        nsims_valid (_type_): _description_
        batch_size_train (int, optional): _description_. Defaults to 1.
        batch_size_valid (int, optional): _description_. Defaults to 1.
        ndims (int, optional): _description_. Defaults to 2.
        dtype (_type_, optional): _description_. Defaults to np.float64.
        return_datasets (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    train_dataset = LandscapeSimulationDataset(
        datdir_train, nsims_train, ndims, 
        transform=None, 
        target_transform=None,
        dtype=dtype,
    )

    valid_dataset = LandscapeSimulationDataset(
        datdir_valid, nsims_valid, ndims, 
        transform=None, 
        target_transform=None,
        dtype=dtype,
    )

    train_dataloader = NumpyLoader(
        train_dataset, 
        batch_size=batch_size_train, 
        shuffle=shuffle_train,
    )

    valid_dataloader = NumpyLoader(
        valid_dataset, 
        batch_size=batch_size_valid, 
        shuffle=shuffle_valid,
    )

    if return_datasets:
        return train_dataloader, valid_dataloader, train_dataset, valid_dataset
    else:
        return train_dataloader, valid_dataloader
    