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
    
    def __init__(
            self, 
            datdir=None, 
            nsims=None, 
            ndims=2, 
            transform=None, 
            target_transform=None, 
            data=None,  # List containing datapoints for each simulation.
            ncells_sample=None,
            dtype=torch.float32,
            rng=None,
            seed=None,
            **kwargs
    ):
        #~~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~#
        simprefix = kwargs.get('simprefix', 'sim')
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if rng is None:
            self.rng = np.random.default_rng(seed=seed)
        else:
            self.rng = rng
        self.dtype = dtype
        self.nsims = nsims
        self.ndims = ndims
        self.transform = transform
        self.target_transform = target_transform
        self.ncells_sample = ncells_sample
        self._constant_ncells = None
        if data is None:
            # Load from given directory
            self._load_data(datdir, nsims, simprefix=simprefix)
        else:
            # Load given data directly
            self.nsims = nsims if nsims else len(data)
            self._load_data_direct(data, nsims)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        t0, x0, t1, x1, sigparams = data

        if self.constant_ncells():
            pass
        else:
            x0 = self.get_subsample(x0, self.ncells_sample)
            x1 = self.get_subsample(x1, self.ncells_sample)
        
        # Transform input x
        if self.transform == 'tensor':
            x = np.concatenate([[t0], [t1], x0.flatten(), sigparams.flatten()])
            x = torch.tensor(x, dtype=self.dtype)
        elif self.transform:
            x = self.transform(*data)
        else:
            x = t0, x0, t1, sigparams
        
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
        t0, x0, t1, x1, sigparams = data
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
        # s = f"$t:{t0:.4g}\\to{t1:.4g}$\
        #     \n$t^*={sigparams[0]:.3g}$\
        #     \n$p_0=[{sigparams[1]:.3g}, {sigparams[2]:.3g}]$\
        #     \n$p_1=[{sigparams[3]:.3g}, {sigparams[4]:.3g}]$"
        s = f"$t:{t0:.4g}\\to{t1:.4g}$\n"
        s += "\n".join([f"$s_{i+1}: {', '.join([f'{x:.3g}' for x in p])}$" for i, p in enumerate(sigparams)])
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
        # ps_all = []
        dataset = []
        for i in range(nsims):
            dirpath = f"{datdir}/{simprefix}{i}"
            assert os.path.isdir(dirpath), f"Not a directory: {dirpath}"
            ts = np.load(f"{dirpath}/ts.npy", allow_pickle=True)
            xs = np.load(f"{dirpath}/xs.npy", allow_pickle=True)
            # ps = np.load(f"{dirpath}/ps.npy", allow_pickle=True)
            sigparams = np.load(f"{dirpath}/sigparams.npy")
            ts_all.append(ts)
            xs_all.append(xs)
            # ps_all.append(ps)
            self._add_sim_data_to_dataset(dataset, ts=ts, xs=xs, ps=None, #ps=ps, 
                                          sigparams=sigparams)
        self.dataset = np.array(dataset, dtype=object)
        self.ts_all = ts_all
        self.xs_all = xs_all
        # self.ps_all = ps_all

    def _add_sim_data_to_dataset(self, dataset, ts, xs, ps, sigparams):
        ntimes = len(ts)
        for i in range(ntimes - 1):
            x0, x1 = xs[i], xs[i + 1]
            t0, t1 = ts[i], ts[i + 1]
            dataset.append((t0, x0, t1, x1, sigparams))

    def _load_data_direct(self, data, nsims):
        assert len(data) == nsims, \
            f"Data (length {len(data)}) should be of length nsims={nsims}."
        dataset = []
        for i in range(nsims):
            sigparams, simdata = data[i]
            sigparams = np.array(sigparams, dtype=float)
            for datapoint in simdata:
                t0 = np.array(datapoint['t0'], dtype=float)
                x0 = np.array(datapoint['x0'], dtype=float)
                t1 = np.array(datapoint['t1'], dtype=float)
                x1 = np.array(datapoint['x1'], dtype=float)
                dataset.append((t0, x0, t1, x1, sigparams))

        self.dataset = np.array(dataset, dtype=object)
        self.ts_all = None
        self.xs_all = None
        self.ps_all = None

    def constant_ncells(self):
        if self._constant_ncells is not None:
            return self._constant_ncells
        else:
            nc_x0s = [len(self.dataset[i][1]) for i in range(len(self))]
            nc_x1s = [len(self.dataset[i][3]) for i in range(len(self))]
            all_same = np.all(np.array(nc_x0s + nc_x1s) == nc_x0s[0])
            self._constant_ncells = all_same
            return all_same
        
    def get_subsample(self, x, ncells):
        ncells_input, dim = x.shape
        x_samp = np.nan * np.ones([ncells, dim], dtype=self.dtype)
        if ncells_input < ncells:
            # Sample with replacement
            samp_idxs = jnp.array(
                self.rng.choice(ncells_input, ncells, True),
                dtype=int,
            )
            x_samp[:] = x[samp_idxs]
        else:
            # Sample without replacement
            samp_idxs = jnp.array(
                self.rng.choice(ncells_input, ncells, False),
                dtype=int,
            )
            x_samp[:] = x[samp_idxs]
        return x_samp
        

###############################
##  Custom DataLoader Class  ##
###############################

# def custom_collate(batch):
#     xs = [dp[0][1] for dp in batch] + [dp[1] for dp in batch]
#     same_dims = np.all(np.array([x.shape for x in xs]) == xs[0].shape)
#     if same_dims:  # Fall back to `default_collate`
#         return default_collate(batch)
#     else:  # Some custom condition
#         t0s = [dp[0][0] for dp in batch]
#         x0s = [dp[0][1] for dp in batch]
#         t1s = [dp[0][2] for dp in batch]
#         sigparams = [dp[0][3] for dp in batch]
#         x1s = [dp[1] for dp in batch]
#         inputs = (
#             t0s, 
#             x0s, 
#             t1s, 
#             sigparams
#         )
#         outputs = x1s
#         return inputs, outputs

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
        batch_size_test=1,
        ndims=2,
        dtype=np.float64,
        shuffle_train=True,
        shuffle_valid=False,
        shuffle_test=False,
        return_datasets=False,
        include_test_data=False,
        datdir_test="",
        nsims_test=None,
        ncells_sample=None,
        rng=None,
        seed=None,
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
    if rng is None:
        rng = np.random.default_rng(seed=seed)

    train_dataset = LandscapeSimulationDataset(
        datdir_train, nsims_train, ndims, 
        transform=None, 
        target_transform=None,
        dtype=dtype,
        ncells_sample=ncells_sample,
        seed=rng.integers(2**32)
    )

    valid_dataset = LandscapeSimulationDataset(
        datdir_valid, nsims_valid, ndims, 
        transform=None, 
        target_transform=None,
        dtype=dtype,
        ncells_sample=ncells_sample,
        seed=rng.integers(2**32),
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

    return_datasets = [train_dataset, valid_dataset]
    return_dataloaders = [train_dataloader, valid_dataloader]

    if include_test_data:
        test_dataset = LandscapeSimulationDataset(
            datdir_test, nsims_test, ndims, 
            transform=None, 
            target_transform=None,
            dtype=dtype,
            ncells_sample=ncells_sample,
            seed=rng.integers(2**32)
        )
        test_dataloader = NumpyLoader(
            test_dataset, 
            batch_size=batch_size_test, 
            shuffle=shuffle_test,
        )
        return_datasets.append(test_dataset)
        return_dataloaders.append(test_dataloader)

    if return_datasets:
        return tuple(return_dataloaders + return_datasets)
    else:
        return tuple(return_dataloaders)
    
    