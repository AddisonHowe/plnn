import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PIL import Image
import time
import matplotlib.animation as animation

import warnings
warnings.filterwarnings("ignore", module="matplotlib\..*")

"""

"""

class PhiSimulationAnimator:
    """Animation handler for landscape simulations.
    """

    savegif = True
    fps = 2                 # default frames per second
    dpi = 100               # default resolution
    interval = 200          # default delay between frames in milliseconds.
    figsize = (12, 9)       # default figure size

    _main_scatter_color = 'k'
    _siglinecmap = 'tab10'
    _paramlinecmap = 'tab10'
    _biflinecmap = 'tab10'
    _axlimbuffer = 0.05  # percent buffer on axis lims

    def __init__(
            self, 
            ts, 
            xys, 
            sigs,
            ps,
            ts_saved, 
            xys_saved, 
            sigs_saved,
            ps_saved, 
            xlims=None,
            ylims=None,
            p0lims=None,
            p1lims=None,
            p0idx=0,
            p1idx=1,
            phi_func = None,
            bifcurves=None,
            bifcolors=None,
            grads=None,
            grad_func=None,
            info_str="",
            sigparams_str="",
            sig_names=None,
            param_names=None,
    ):
        """
        ts         : (nts,)
        xys        : (nts, ncells, ndims)
        sigs       : (nts, nsigs)
        ps         : (nts, nparams)
        ts_saved   : (nsaves)
        xys_saved  : (nsaves, ncells, ndims)
        sigs_saved : (nsaves, nsigs)
        ps_saved   : (nsaves, nparams)
        sigparams  : (nsaves, nsigs, nsigparams)
        xlims : TODO
        ylims : TODO
        grads : TODO
        grad_func : TODO
        """
        self.ts = ts
        self.xys = xys
        self.xs = xys[:,:,0]
        self.ys = xys[:,:,1]
        self.sigs = sigs
        self.nsigs = sigs.shape[1]
        self.ps = ps
        self.nparams = ps.shape[1]
        self.p0s = ps[:,p0idx]
        self.p1s = ps[:,p1idx]
        self.p0idx = p0idx
        self.p1idx = p1idx
        self.phi_func = phi_func
        self.bifcurves = bifcurves
        self.bifcolors = bifcolors
        self.info_str = info_str
        self.sigparams_str = sigparams_str
        if sig_names is None:
            self.sig_names = [f'$s_{i}$' for i in range(self.nsigs)]
        else:
            self.sig_names = sig_names
        if param_names is None:
            self.param_names = [f'$p_{i}$' for i in range(self.nparams)] 
        else:
            self.param_names = param_names

        self.ts_saved = ts_saved
        self.xys_saved = xys_saved
        self.xs_saved = xys_saved[:,:,0]
        self.ys_saved = xys_saved[:,:,1]
        self.sigs_saved = sigs_saved
        self.ps_saved = ps_saved
        self.p0s_saved = ps_saved[:,p0idx]
        self.p1s_saved = ps_saved[:,p1idx]

        self.grads = grads
        self.grad_func = grad_func
        
        self.nframes = len(self.ts)
        self.nsaves = len(self.ts_saved)

        self.xlims = xlims if xlims else self._buffer_lims(
            [np.min(self.xs), np.max(self.xs)]
        )
        self.ylims = ylims if ylims else self._buffer_lims(
            [np.min(self.ys), np.max(self.ys)]
        )
        
        self.p0lims = p0lims if p0lims else self._buffer_lims(
            [np.min(self.p0s), np.max(self.p0s)]
        )
        self.p1lims = p1lims if p1lims else self._buffer_lims(
            [np.min(self.p1s), np.max(self.p1s)]
        )

        # Mesh for heatmap
        heatmap_n = 50
        self.heatmeshx, self.heatmeshy = np.meshgrid(
            np.linspace(*self.xlims, heatmap_n),
            np.linspace(*self.ylims, heatmap_n)
        )
        xy = np.array([self.heatmeshx.ravel(), self.heatmeshy.ravel()]).T
        
        phis = np.array(
            [self.phi_func(t, xy, self.ps[i]) for i, t in enumerate(self.ts)]
        )
        self.phis = np.log(1 + phis - phis.min())  # log normalize


    def animate(self, savepath=None, **kwargs):
        """"""
        #~~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~#
        duration = kwargs.get('duration', None)
        interval = kwargs.get('interval', self.interval)
        figsize = kwargs.get('figsize', self.figsize)
        dpi = kwargs.get('dpi', self.dpi)
        fps = kwargs.get('fps', self.fps)
        save_frames = kwargs.get('save_frames', [])
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

        if duration:
            fps = self.nframes / duration
            interval = 1000 / fps
        
        self.fig = plt.figure(
            dpi=dpi, 
            figsize=figsize, 
            constrained_layout=True,
        )
        
        self.ax_main = plt.subplot2grid((4, 6), (0, 0), colspan=2, rowspan=2)
        self.ax_clst = plt.subplot2grid((4, 6), (2, 0), colspan=2, rowspan=2)
        self.ax_sigs = plt.subplot2grid((4, 6), (0, 2), colspan=2, rowspan=1)
        self.ax_prms = plt.subplot2grid((4, 6), (1, 2), colspan=2, rowspan=1)
        self.ax_heat = plt.subplot2grid((4, 6), (2, 2), colspan=2, rowspan=2)
        self.ax_bifs = plt.subplot2grid((4, 6), (0, 4), colspan=2, rowspan=2)
        self.ax_text = plt.subplot2grid((4, 6), (2, 4), colspan=2, rowspan=2)

        print("Generating movie...", flush=True)
        tic0 =time.time()
        self.ani = animation.FuncAnimation(
            self.fig, 
            self.update, 
            frames=self.nframes, 
            interval=interval, 
			init_func=self.setup, 
            blit=True
        )
        tic1 = time.time() 
        print(f"Finished in {tic1-tic0:.3g} seconds.", flush=True)
        if savepath:
            print(f"Saving animation to {savepath}", flush=True)
            if self.savegif:
                fpath = savepath+'.gif'
                self.ani.save(fpath, writer='pillow', fps=fps)
                frames_to_save = save_frames
                with Image.open(fpath) as im:
                    for frameidx in frames_to_save:
                        im.seek(frameidx)
                        im.save(f'{savepath}_frame{frameidx}.png')
            else:
                fpath = savepath+'.mp4'
                self.ani.save(fpath, fps=fps)
        return self.ani

    def setup(self):
        """Initialize each axis and text."""
        self._setup_main()
        self._setup_clst()
        self._setup_sigs()
        self._setup_prms()
        self._setup_heat()
        self._setup_bifs()
        self._setup_text()
        return (
            self.scat_main, self.scat_clst, *self._signal_markers,
            *self._param_markers, self._bif_marker, self.heatmap
        )
        
    def update(self, i):
        """Update each axis and text."""
        self._update_main(i)
        self._update_clst(i)
        self._update_sigs(i)
        self._update_prms(i)
        self._update_heat(i)
        self._update_bifs(i)
        self._update_text(i)
        return (
            self.scat_main, self.scat_clst, *self._signal_markers,
            *self._param_markers, self._bif_marker, self.heatmap
        )

    #####################
    ##  Setup Methods  ##
    #####################
        
    def _setup_main(self):
        ax = self.ax_main
        # Initialize
        self.scat_main, = ax.plot(
            [], [], 
            marker='.', 
            markersize=4, 
            color=self._main_scatter_color, 
            alpha=0.75, 
            linestyle='None', 
            animated=True
        )

        # Format
        ax.axis([*self.xlims, *self.ylims])
        title = f""
        ax.set_title(title)
        ax.set_xlabel(f"$x$")
        ax.set_ylabel(f"$y$")
        
        # Text
        pos = [self.xlims[0] + (self.xlims[1]-self.xlims[0])*.5, 
               self.ylims[0] + (self.ylims[1]-self.ylims[0])*.95]
        self.maintext = ax.text(*pos, "", fontsize='small')
        
        # Mesh for gradient
        self.meshx, self.meshy = np.meshgrid(np.linspace(*self.xlims, 20),
                                             np.linspace(*self.ylims, 20))
        self.mesh_gradient = ax.quiver(
            self.meshx, 
            self.meshy, 
            np.zeros(self.meshx.shape), 
            np.zeros(self.meshy.shape),
            color=self._main_scatter_color, 
            alpha=0.5, 
            animated=True
        )

        self.heatmap = ax.pcolormesh(
            self.heatmeshx, 
            self.heatmeshy, 
            np.zeros(self.heatmeshx.shape), 
            vmin=self.phis.min(),
            vmax=self.phis.max(),
            cmap='coolwarm', 
            animated=True,
            shading='gouraud',
        )

    def _setup_clst(self):
        ax = self.ax_clst
        self.scat_clst, = ax.plot(
            [], [], marker='.', markersize=4, 
            color=self._main_scatter_color, 
            alpha=0.75, linestyle='None', 
            animated=True
        )
        self.clst_index = 0
        ax.set_xlabel(f"$x$")
        ax.set_ylabel(f"$y$")
        ax.axis([*self.xlims, *self.ylims])
        # Text
        pos = [self.xlims[0] + (self.xlims[1]-self.xlims[0])*.5, 
               self.ylims[0] + (self.ylims[1]-self.ylims[0])*.95]
        self.clsttext = ax.text(*pos, "", fontsize='small')

    def _setup_sigs(self):
        ax = self.ax_sigs
        # Plot signals
        siglines = []
        # cmap = plt.cm.get_cmap(self._siglinecmap)
        cmap = plt.colormaps[self._siglinecmap]
        for i in range(self.nsigs):
            line, = ax.plot(
                self.ts, self.sigs[:,i], 
                label=self.sig_names[i], 
                color=cmap(i),
            )
            siglines.append(line)
        ax.legend(siglines, self.sig_names)
        # Add marker placeholders
        self._signal_markers = []
        for i in range(self.nsigs):
            marker, = ax.plot(
                [], [], 
                marker='*', 
                markersize=6, color='k', alpha=0.9, 
                linestyle='None', animated=True
            )
            self._signal_markers.append(marker)
        # Add vertical lines
        ylims = ax.get_ylim()
        ax.vlines(self.ts_saved, *ylims, linestyle=':', colors='k', alpha=0.25)
        # Axis labeling
        ax.set_xlabel(f"$t$")
        ax.set_ylabel(f"$s$")

    def _setup_prms(self):
        ax = self.ax_prms
        # Plot signals
        paramlines = []
        # cmap = plt.cm.get_cmap(self._paramlinecmap)
        cmap = plt.colormaps[self._paramlinecmap]
        for i in range(self.nparams):
            line, = ax.plot(
                self.ts, self.ps[:,i], 
                label=self.param_names[i], 
                color=cmap(i),
            )
            paramlines.append(line)
        ax.legend(paramlines, self.param_names)
        # Add marker placeholders
        self._param_markers = []
        for i in range(self.nparams):
            marker, = ax.plot(
                [], [], 
                marker='*', 
                markersize=6, color='k', alpha=0.9, 
                linestyle='None', animated=True
            )
            self._param_markers.append(marker)
        # Add vertical lines
        ylims = ax.get_ylim()
        ax.vlines(self.ts_saved, *ylims, linestyle=':', colors='k', alpha=0.25)
        # Axis labeling
        ax.set_xlabel(f"$t$")
        ax.set_ylabel(f"$p$")

    def _setup_heat(self):
        ax = self.ax_heat
        # self.scat_dist, = ax.plot(
        #     [], [], marker='.', markersize=4, 
        #     color=self._main_scatter_color, 
        #     alpha=0.75, linestyle='None', 
        #     animated=True
        # )
        self.dist_index = 0
        ax.set_xlabel(f"$x$")
        ax.set_ylabel(f"$y$")
        ax.axis([*self.xlims, *self.ylims])
        # Text
        pos = [self.xlims[0] + (self.xlims[1]-self.xlims[0])*.5, 
               self.ylims[0] + (self.ylims[1]-self.ylims[0])*.95]
        self.disttext = ax.text(*pos, "", fontsize='small')

    def _setup_bifs(self):
        ax = self.ax_bifs
        ax.axis([*self.p0lims, *self.p1lims])
        # Plot bifurcation curves
        if self.bifcurves is not None:
            self._plot_bifcurves(ax)
        # Plot trace of parameter values
        ax.plot(
            self.p0s, self.p1s, alpha=0.5, linestyle=':', color='k'
        )
        # Add marker placeholders
        self._bif_marker, = ax.plot(
            [], [], 
            marker='*', 
            markersize=6, color='k', alpha=0.9, 
            linestyle='None', animated=True
        )
        # Axis labeling
        ax.set_xlabel(self.param_names[self.p0idx])
        ax.set_ylabel(self.param_names[self.p1idx])

    def _setup_text(self):
        ax = self.ax_text
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        pos = [xlims[0] + (xlims[1]-xlims[0])*.2, 
               ylims[0] + (ylims[1]-ylims[0])*0.05]
        self.text = ax.text(*pos, "", fontsize='small')

    ######################
    ##  Update Methods  ##
    ######################

    def _update_main(self, i):
        ax = self.ax_main
        xy = self.get_xy(i)
        self.scat_main.set_data(xy)
        self.maintext.set_text(f"t={self.get_t(i):.3f}")
        grads = self.get_grads_mesh(i)
        if grads is not None:
            u, v = grads
            norms = np.sqrt(u*u + v*v)
            u = u/norms
            v = v/norms
            self.mesh_gradient.set_UVC(u, v)
            pcm = plt.get_cmap('RdBu_r')
            cnorm = colors.Normalize(vmin=0, vmax=np.max(norms))
            self.mesh_gradient.set_color(pcm(cnorm(norms).flatten()))
        # Update heatmap
        phi = self.phis[i]
        self.heatmap.set_array(phi.ravel())

    def _update_clst(self, i):
        ax = self.ax_clst
        if self.ts_saved is None:
            return 
        t = self.get_t(i)
        clst_t = self.ts_saved[self.clst_index]
        if t == clst_t:
            xy_saved = self.get_xy_saved(self.clst_index)
            self.clst_index += 1
            self.scat_clst.set_data(xy_saved)

    def _update_sigs(self, i):
        t = self.get_t(i)
        sigs = self.get_sigs(i)
        for j, sig in enumerate(sigs):
            self._signal_markers[j].set_data([[t], [sig]])

    def _update_prms(self, i):
        t = self.get_t(i)
        params = self.get_ps(i)
        for j, param in enumerate(params):
            self._param_markers[j].set_data([[t], [param]])

    def _update_heat(self, i):
        ax = self.ax_heat
        t = self.get_t(i)
        dist_t = self.ts_saved[self.dist_index]
        if t == dist_t:
            xy_saved = self.get_xy_saved(self.dist_index)
            self.dist_index += 1
            ax.plot(
                *xy_saved,
                marker='.', markersize=4, 
                alpha=0.75, linestyle='None',
            )

    def _update_bifs(self, i):
        params = self.get_ps(i)
        p0 = params[self.p0idx]
        p1 = params[self.p1idx]
        self._bif_marker.set_data([[p0], [p1]])

    def _update_text(self, i):
        t = self.get_t(i)
        s = self.info_str + f"\n$t={t:.3f}$\n" + self.sigparams_str
        self.text.set_text(s)

    ######################
    ##  Getter Methods  ##
    ######################

    def get_t(self, i):
        return self.ts[i]
    
    def get_xy(self, i):
        return self.xys[i, :].T
    
    def get_sigs(self, i):
        return self.sigs[i,:]
    
    def get_ps(self, i):
        return self.ps[i,:]
    
    def get_t_saved(self, i):
        return self.ts_saved[i]

    def get_xy_saved(self, i):
        return self.xys_saved[i, :].T
    
    def get_sigs_saved(self, i):
        return self.sigs_saved[i,:]
    
    def get_ps_saved(self, i):
        return self.ps_saved[i,:]
    
    ######################
    ##  Helper Methods  ##
    ######################
    
    def _buffer_lims(self, lims):
        buffer = self._axlimbuffer * (lims[1] - lims[0])
        return [lims[0] - buffer, lims[1] + buffer]
        
    def get_grads_mesh(self, i):
        if self.grads is not None:
            return self.grads[i]
        elif self.grad_func:
            x = self.meshx.flatten()
            y = self.meshy.flatten()
            p = self.get_sigs(i)
            gradx, grady = self.grad_func(x, y, p)
            return gradx, grady
        else:
            return None

    def _plot_bifcurves(self, ax, linestyle='--'):
        ncurves = len(self.bifcurves)
        for i in range(ncurves):
            color = self.bifcolors[i]
            curve = self.bifcurves[i]
            ax.plot(curve[:,0], curve[:,1], color=color, linestyle=linestyle)
