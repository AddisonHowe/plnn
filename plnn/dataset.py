"""Custom Dataset and DataLoader classes.
"""

import os
import warnings
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
    """A collection of data representing consecutive snapshots of a system.
    
    The data consists of tuples (t_i, X_i, t_j, X_j, <args>), where t_i and t_j
    are scalar floats, and X_i and X_j are matrices of size [?, d]. Thus, the
    state dimension is fixed, but the number of cells at the two timepoints may
    differ. If the number of cells is the same across the entire dataset, then 
    the dataset is "homogeneous."

    Samples may be drawn from a dataset by directly sampling the datapoint 
    tuples. Alternatively, an additional step of subsampling can be performed.
    In that case, a specified number of cells are sampled from the X_i and X_j
    matrices whenever an item is queried via __getitem__.
    """

    nonhomogeneous_warning = "Dataset is not homogeneous."
    no_sample_with_ncells_warning = \
        "Must set transform='sample' if ncells_sample > 0."
    
    def __init__(
            self, 
            datdir=None, 
            nsims=None, 
            ndims=2, 
            transform=None, 
            data=None,  # List containing datapoints for each simulation.
            ncells_sample=0,
            length_multiplier=1,
            rng=None,
            seed=None,
            simprefix='sim',
            suppress_warnings=False,
    ):
        if rng is None:
            self.rng = np.random.default_rng(seed=seed)
        else:
            self.rng = rng
        self.nsims = nsims
        self.ndims = ndims
        self.transform = transform
        self.ncells_sample = ncells_sample
        self.length_multiplier = length_multiplier
        assert isinstance(ncells_sample, int), \
            f"Got ncells_sample={ncells_sample}"
        
        # Load data directly or from given directory
        if data is None:  
            self._load_data(datdir, nsims, simprefix=simprefix)
        else:
            self.nsims = nsims if nsims else len(data)
            self._load_data_direct(data, nsims)
        
        # Check if each datapoint has the same number of cells.
        self._is_loadable = True
        self.is_homogeneous = self._check_homogeneous()
        if ncells_sample > 0 and self.transform != 'sample':
            msg = self.no_sample_with_ncells_warning
            raise RuntimeError(msg)
        if not self.is_homogeneous:
            msg = self.nonhomogeneous_warning
            if self.transform != "sample":
                self._is_loadable = False
                msg += " Need to set transform='sample' and" 
                msg += " ncells_sample=<int> to use NumpyLoader."
                msg += " Set suppress_warnings=False to suppress this message."
                if not suppress_warnings:
                    warnings.warn(RuntimeWarning(msg))
            elif self.ncells_sample == 0:
                msg += " Need to set ncells_sample > 0."
                raise RuntimeError(msg)

    def __len__(self):
        return len(self.dataset) * self.length_multiplier
    
    def __getitem__(self, idx):
        if idx >= len(self):
            msg = f"Index {idx} out of bound for {type(self)}."
            msg += f" Dataset has length {len(self.dataset)} "
            msg += f" with multiplier {self.length_multiplier}."
            raise IndexError(msg)
        idx = idx % len(self.dataset)
        data = self.dataset[idx]
        t0, x0, t1, x1, sigparams = data
        # Sample if needed
        if self.transform == 'sample':
            # TODO: different ncells_sample for x0 and x1?
            x0 = self.get_subsample(x0, self.ncells_sample)
            x1 = self.get_subsample(x1, self.ncells_sample)
        inputs = t0, x0, t1, sigparams
        outputs = x1
        return inputs, outputs
    
    def get_unsampled_item(self, idx):
        if self.is_homogeneous:
            return self[idx]  # NOTE: untested
        data = self.dataset[idx]
        t0, x0, t1, x1, sigparams = data
        inputs = t0, x0, t1, sigparams
        outputs = x1
        return inputs, outputs
    
    def get_all_cells(self):
        return np.array(self.xs_all)
        
    def get_baselength(self):
        return len(self.dataset)
    
    def is_loadable(self):
        return self._is_loadable

    def is_not_loadable(self):
        return not self._is_loadable
    
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
        title = kwargs.get('title', f"datapoint {idx}/{len(self.dataset)}")
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
        ax.set_title(title)
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

    def animate(self, simidx, interval=50, saveas=False, fps=1, **kwargs):
        """Animate a given simulation"""
        idx0 = int(simidx * len(self.dataset) // self.nsims)
        idx1 = idx0 + int(len(self.dataset) // self.nsims)
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
        if saveas:
            anim.save(saveas, fps=fps)
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

    def _check_homogeneous(self):
        nc_x0s = [len(self.dataset[i][1]) for i in range(len(self.dataset))]
        nc_x1s = [len(self.dataset[i][3]) for i in range(len(self.dataset))]
        all_same = np.all(np.array(nc_x0s + nc_x1s) == nc_x0s[0])
        return all_same

    def get_subsample(self, x, ncells):
        ncells_input, dim = x.shape
        x_samp = np.nan * np.ones([ncells, dim], dtype=float)
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

    nonloadable_error_message = "NumpyLoader given nonloadable dataset."

    def __init__(
            self, 
            dataset: LandscapeSimulationDataset, 
            batch_size=1,
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
        if dataset.is_not_loadable():
            msg = self.nonloadable_error_message
            raise RuntimeError(msg)


def get_dataloaders(
        datdir_train,
        datdir_valid,
        nsims_train,
        nsims_valid,
        batch_size_train=1,
        batch_size_valid=1,
        batch_size_test=1,
        ndims=2,
        shuffle_train=True,
        shuffle_valid=False,
        shuffle_test=False,
        return_datasets=False,
        include_test_data=False,
        datdir_test="",
        nsims_test=None,
        ncells_sample=0,
        length_multiplier=1,
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

    transform = "sample" if ncells_sample else None
    train_dataset = LandscapeSimulationDataset(
        datdir_train, nsims_train, ndims, 
        transform=transform, 
        ncells_sample=ncells_sample,
        length_multiplier=length_multiplier,
        seed=rng.integers(2**32)
    )

    valid_dataset = LandscapeSimulationDataset(
        datdir_valid, nsims_valid, ndims, 
        transform=transform, 
        ncells_sample=ncells_sample,
        length_multiplier=length_multiplier,
        seed=rng.integers(2**32),
    )

    train_dataloader = NumpyLoader(
        train_dataset, 
        batch_size=min(batch_size_train, len(train_dataset)), 
        shuffle=shuffle_train,
    )

    valid_dataloader = NumpyLoader(
        valid_dataset, 
        batch_size=min(batch_size_valid, len(valid_dataset)), 
        shuffle=shuffle_valid,
    )

    return_datasets = [train_dataset, valid_dataset]
    return_dataloaders = [train_dataloader, valid_dataloader]

    if include_test_data:
        test_dataset = LandscapeSimulationDataset(
            datdir_test, nsims_test, ndims, 
            transform=transform, 
            ncells_sample=ncells_sample,
            length_multiplier=length_multiplier,
            seed=rng.integers(2**32)
        )
        test_dataloader = NumpyLoader(
            test_dataset, 
            batch_size=min(batch_size_test, len(test_dataset)), 
            shuffle=shuffle_test,
        )
        return_datasets.append(test_dataset)
        return_dataloaders.append(test_dataloader)

    if return_datasets:
        return tuple(return_dataloaders + return_datasets)
    else:
        return tuple(return_dataloaders)
    
    