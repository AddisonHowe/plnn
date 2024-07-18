"""

"""

import numpy as np
import matplotlib as mpl
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from PIL import Image
import time
import matplotlib.animation as animation


from plnn.models.plnn import PLNN
from plnn.pl import DEFAULT_CMAP, CHIR_COLOR, FGF_COLOR

import warnings
warnings.filterwarnings("ignore", module="matplotlib\..*")


class PLNNSimulationAnimator:
    """Animation handler for PLNN simulations.
    """

    savegif = True
    figsize = (12, 9)       # default figure size

    _main_scatter_size = 5
    _main_scatter_color = 'k'
    _mins_scatter_size = 6
    _mins_scatter_color = 'y'
    _mins_scatter_alpha = 0.5
    _clst_scatter_color1 = 'r'
    _clst_scatter_color2 = 'k'
    _surf_scatter_size = 5
    _surf_scatter_color = 'k'
    _siglinecmap = 'signal_cmap'
    _paramlinecmap = 'Accent'
    _biflinecmap = 'tab10'
    _axlimbuffer = 0.05  # percent buffer on axis lims
    _bifcurvecolor = 'r'
    _linewidth = 2
    _surface_alpha = 1.0
    _gradient_field_cmap = 'RdBu_r'
    _scatter_fixed_point_markers = {
        'saddle': 'x',
        'minimum': '*',
        'maximum': 'o',
    }
    
    _font_scale_factor = 1
    _suptitlesize    = 8 * _font_scale_factor
    _titlesize       = 7 * _font_scale_factor
    _axlabelsize     = 6 * _font_scale_factor
    _axmajorticksize = 5 * _font_scale_factor
    _axminorticksize = 4 * _font_scale_factor
    _legendsize      = 5 * _font_scale_factor
    _rtext_fontsize  = 6 * _font_scale_factor
    _btext_fontsize  = 6 * _font_scale_factor


    def __init__(
            self, 
            model : PLNN, 
            sigparams,
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
            zlims=None,
            p0lims=None,
            p1lims=None,
            p0idx=0,
            p1idx=1,
            minima=None,
            fixed_point_info=None,
            bifcurves=None,
            bifcolors=None,
            grads=None,
            grad_func=None,
            note_string="",
            sigparams_str="",
            sig_names=None,
            param_names=None,
            view_init=(40, -45),  # elevation, azimuthal
            observed_data=None,
            losses=None,
    ):
        """
        model      : PLNN
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
        self.model = model
        self.sigparams = sigparams
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
        self.phi_func = model.tilted_phi
        self.minima = minima
        self.fixed_point_info = fixed_point_info
        self.bifcurves = bifcurves
        self.bifcolors = bifcolors
        self.note_string = note_string
        self.sigparams_str = sigparams_str
        if sig_names is None:
            self.sig_names = [f'$s_{i}$' for i in range(self.nsigs)]
        else:
            self.sig_names = sig_names
        if param_names is None:
            self.param_names = [f'$\\tau_{i}$' for i in range(self.nparams)] 
        else:
            self.param_names = param_names
        if self.nsigs == 2:
            # self._siglinecmap = [CHIR_COLOR, FGF_COLOR]
            pass

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

        self.observed_data = observed_data
        self.show_sim_vs_obs = observed_data is not None
        self.losses = losses
        
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
        heatmap_xy = np.array([self.heatmeshx.ravel(),
                               self.heatmeshy.ravel()]).T
        heatmap_phis = np.array(
            [self.phi_func(t, heatmap_xy, self.sigparams) for t in self.ts]
        )
        minphi = heatmap_phis.min()
        self.heatmap_phis = np.log(1 + heatmap_phis - minphi)

        # 3D scatterplot phi values
        # NOTE: Cannot currently implement 3d scatterplot due to mpl issue.
        phis = np.array(
            [self.phi_func(t, xy, self.sigparams) 
             for t, xy in zip(self.ts, self.xys)]
        )
        self.phis = np.log(1 + phis - minphi) + 3e-1


        self.zlims = zlims if zlims else self._buffer_lims(
            [np.min(self.heatmap_phis), np.max(self.heatmap_phis)]
        )

        self.view_init = view_init

        self.model_info_string = model.get_info_string()
        self._rtext_list = self._generate_rtext_list()
        self._btext_list = self._generate_btext_list()


    def animate(
            self, 
            duration=None,
            interval=200,
            fps=2,
            figsize=None,
            grid_width=2,
            grid_height=2,
            dpi=100,
            save_frames=[],
            saveas='gif',
            savepath=None,
            suptitle="Cellular Decisions",
    ):
        """Generate animation.
        
        Args:
            duration (int) : total time of animation in seconds. If specified, 
                overrides `fps` and `interval` arguments.
            interval (int) : default delay between frames in milliseconds.
            fps (int) : number of frames per second.
            figsize (tuple[float]) : figure width and height. If specified, 
                overrides `grid_width` and `grid_height` arguments.
            grid_width (float) : width of each grid element.
            grid_height (float) : height of each grid element.
            dpi (int) : dots per inch resolution.
            save_frames (list[int]) : indices of frames to save as png files.
            saveas (str) : File type of saved movie. Options are 'gif' or 'mp4'.
                Default 'gif'.
            savepath (str) : File path to save the movie to, without extension.
        Returns:
            Animation
        """

        if duration:
            fps = self.nframes / duration
            interval = 1000 / fps
        
        grdsz = (7, 8)  # number of grid rows and columns
        
        if figsize is None:
            figsize = (grid_width * grdsz[1], grid_height * grdsz[0])

        self.fig = plt.figure(
            dpi=dpi, 
            figsize=figsize, 
            constrained_layout=True,
        )
        
        self.ax_main = plt.subplot2grid(grdsz, (0, 0), colspan=3, rowspan=3)
        self.ax_surf = plt.subplot2grid(grdsz, (0, 3), colspan=3, rowspan=3,
                                        projection='3d')
        self.ax_rmisc = plt.subplot2grid(grdsz, (0, 6), colspan=2, rowspan=5)
        self.ax_clst = plt.subplot2grid(grdsz, (3, 0), colspan=2, rowspan=2)
        self.ax_sigs = plt.subplot2grid(grdsz, (3, 2), colspan=2, rowspan=1)
        self.ax_tilt = plt.subplot2grid(grdsz, (4, 2), colspan=2, rowspan=1)
        self.ax_bifs = plt.subplot2grid(grdsz, (3, 4), colspan=2, rowspan=2)
        self.ax_bmisc = plt.subplot2grid(grdsz, (5, 0), colspan=8, rowspan=2)

        self.fig.suptitle(suptitle, size=self._suptitlesize)

        for ax in self.fig.axes:
            ax.tick_params(axis='both', which='major', 
                           labelsize=self._axmajorticksize)
            ax.tick_params(axis='both', which='minor', 
                           labelsize=self._axminorticksize)

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
            if saveas == 'gif':
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
        self._setup_tilt()
        self._setup_surf()
        self._setup_bifs()
        self._setup_rmisc()
        self._setup_bmisc()
        return (
            self.scat_main, self.scat_clst1, *self._signal_markers,
            *self._param_markers, self._bif_marker, self.heatmap, 
            self.gradient_field,
        )
        
    def update(self, i):
        """Update each axis and text."""
        self._update_main(i)
        self._update_clst(i)
        self._update_sigs(i)
        self._update_tilt(i)
        self._update_surf(i)
        self._update_bifs(i)
        self._update_rmisc(i)
        self._update_bmisc(i)
        return (
            self.scat_main, self.scat_clst1, *self._signal_markers,
            *self._param_markers, self._bif_marker, self.heatmap, 
            self.gradient_field,
        )

    #####################
    ##  Setup Methods  ##
    #####################
        
    def _setup_main(self):
        ax = self.ax_main
        ax.set_aspect('equal')
        # Setup main scatterplot
        self.scat_main, = ax.plot(
            [], [], 
            marker='.', 
            markersize=self._main_scatter_size, 
            color=self._main_scatter_color, 
            alpha=0.75, 
            linestyle='None', 
            animated=True
        )
        # Setup minima scatterplot
        if self.minima is not None:
            self.scatter_mins, = ax.plot(
                [], [], 
                marker='*', 
                markersize=self._mins_scatter_size, 
                color=self._mins_scatter_color, 
                alpha=self._mins_scatter_alpha, 
                linestyle='None', 
                animated=True
            )
        elif self.fixed_point_info is not None:
            _, types, fpcolors = self.fixed_point_info
            self.fixed_point_styles = []
            self.scatter_mins = []
            for typeset, colorset in zip(types, fpcolors):
                for t, c in zip(typeset, colorset):
                    if (t, c) not in self.fixed_point_styles:
                        self.fixed_point_styles.append((t, c))
                        new_scatter, = ax.plot(
                            [], [], 
                            marker=self._scatter_fixed_point_markers[t], 
                            markersize=self._mins_scatter_size, 
                            color=c, 
                            alpha=self._mins_scatter_alpha, 
                            linestyle='None', 
                            animated=True
                        )
                        self.scatter_mins.append(new_scatter)

        # Format
        ax.axis([*self.xlims, *self.ylims])
        ax.set_xlabel(f"$x$", size=self._axlabelsize)
        ax.set_ylabel(f"$y$", size=self._axlabelsize)
        ax.set_title("cell simulation", size=self._titlesize)        
        # Text
        pos = [0.05, 0.99]
        self.maintext = ax.text(
            *pos, "", 
            fontsize='small', 
            ha='left', 
            va='top', 
            transform=ax.transAxes
        )
        # Heatmap
        self.heatmap = ax.pcolormesh(
            self.heatmeshx, 
            self.heatmeshy, 
            np.zeros(self.heatmeshx.shape), 
            vmin=self.heatmap_phis.min(),
            vmax=self.heatmap_phis.max(),
            cmap=DEFAULT_CMAP, 
            animated=True,
            shading='gouraud',
        )
        # Gradient field
        self.meshx, self.meshy = np.meshgrid(np.linspace(*self.xlims, 20),
                                             np.linspace(*self.ylims, 20))
        self.gradient_field = ax.quiver(
            self.meshx, 
            self.meshy, 
            np.zeros(self.meshx.shape), 
            np.zeros(self.meshy.shape),
            color=self._main_scatter_color, 
            alpha=0.5, 
            animated=True
        )

    def _setup_surf(self):
        ax = self.ax_surf
        ax.axis([*self.xlims, *self.ylims, *self.zlims])
        
        # Setup surface
        self.surface = [ax.plot_surface(
            self.heatmeshx, 
            self.heatmeshy, 
            np.ones(self.heatmeshx.shape) * np.nan, 
            vmin=self.heatmap_phis.min(),
            vmax=self.heatmap_phis.max(),
            cmap=DEFAULT_CMAP, 
            alpha=self._surface_alpha,
            linewidth=0.1,
            animated=True,
        )]
        # Setup surface scatterplot
        # NOTE: Cannot currently implement 3d scatterplot due to mpl issue.
        # self.surface_scatter, = ax.plot(
        #     [], [], [],
        #     marker='.', 
        #     markersize=self._surf_scatter_size, 
        #     linestyle='None', 
        #     color=self._surf_scatter_color, 
        #     alpha=1.0, 
        #     zorder=2.5,
        #     animated=True
        # )
        ax.view_init(*self.view_init)
        ax.set_xlabel(f"$x$", size=self._axlabelsize)
        ax.set_ylabel(f"$y$", size=self._axlabelsize)
        ax.set_zlabel(f"$\phi$", size=self._axlabelsize)

    def _setup_clst(self):
        ax = self.ax_clst
        ax.set_aspect('equal')
        if self.show_sim_vs_obs:
            label1 = "observed"
            label2 = "simulated"
        else:
            label1 = "previous"
            label2 = "current"
        self.scat_clst1, = ax.plot(
            [], [], marker='.', markersize=self._main_scatter_size, 
            color=self._clst_scatter_color1, 
            alpha=0.25, linestyle='None', 
            animated=True,
            label=label1
        )
        self.scat_clst2, = ax.plot(
            [], [], marker='.', markersize=self._main_scatter_size, 
            color=self._clst_scatter_color2, 
            alpha=0.25, linestyle='None', 
            animated=True,
            label=label2
        )
        self.clst_index = 0
        ax.legend([self.scat_clst1, self.scat_clst2], 
                  [label1, label2], prop={'size': self._legendsize})
        ax.axis([*self.xlims, *self.ylims])
        # Text
        pos = [0.05, 0.01]
        self.clsttext = ax.text(
            *pos, "",
            fontsize='small', 
            ha='left', 
            va='bottom', 
            transform=ax.transAxes,
        )
        # Format
        ax.set_title("snapshots", size=self._titlesize)
        ax.set_xlabel(f"$x$", size=self._axlabelsize)
        ax.set_ylabel(f"$y$", size=self._axlabelsize)

    def _setup_sigs(self):
        ax = self.ax_sigs
        # Plot signals
        siglines = []
        # cmap = plt.cm.get_cmap(self._siglinecmap)
        cmap = plt.colormaps[self._siglinecmap]
        for i in range(self.nsigs):
            line, = ax.plot(
                self.ts, self.sigs[:,i], 
                linewidth=self._linewidth,
                color=cmap(i),
                label=self.sig_names[i], 
            )
            siglines.append(line)
        ax.legend(siglines, self.sig_names, prop={'size': self._legendsize})
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
        ax.set_title("signal", size=self._titlesize)
        ax.set_xlabel(f"$t$", size=self._axlabelsize)
        ax.set_ylabel(f"$s$", size=self._axlabelsize)

    def _setup_tilt(self):
        ax = self.ax_tilt
        # Plot signals
        paramlines = []
        # cmap = plt.cm.get_cmap(self._paramlinecmap)
        cmap = plt.colormaps[self._paramlinecmap]
        for i in range(self.nparams):
            line, = ax.plot(
                self.ts, self.ps[:,i], 
                linewidth=self._linewidth,
                color=cmap(i),
                label=self.param_names[i], 
            )
            paramlines.append(line)
        ax.legend(paramlines, self.param_names, prop={'size': self._legendsize})
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
        ax.set_title("tilts", size=self._titlesize)
        ax.set_xlabel(f"$t$", size=self._axlabelsize)
        ax.set_ylabel(f"$\\tau$", size=self._axlabelsize)

    def _setup_bifs(self):
        ax = self.ax_bifs
        ax.set_aspect('equal')
        ax.axis([*self.p0lims, *self.p1lims])
        # Plot bifurcation curves
        if self.bifcurves is not None:
            self._plot_bifcurves(
                ax, 
                linestyle='-',
                linewidth=1,
            )
        # Plot trace of tilt values
        traj = ax.plot(
            self.p0s, self.p1s, alpha=0.5, linestyle=':', color='k',
        )
        # Add marker placeholders
        self._bif_marker, = ax.plot(
            [], [], 
            marker='*', 
            markersize=6, color='k', alpha=0.9, 
            linestyle='None', animated=True
        )
        # Axis labeling
        ax.legend(traj, ["$\\tau(t)$"], prop={'size': self._legendsize})
        ax.set_xlabel(self.param_names[self.p0idx], size=self._axlabelsize)
        ax.set_ylabel(self.param_names[self.p1idx], size=self._axlabelsize)
        ax.set_title("bifurcation plot", size=self._titlesize)

    def _setup_rmisc(self):
        ax = self.ax_rmisc
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        pos = [xlims[0] + (xlims[1]-xlims[0])*0.01, 
               ylims[0] + (ylims[1]-ylims[0])*0.99]
        self.rtext = ax.text(
            *pos, "", 
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=self._rtext_fontsize,
            usetex=False,
            animated=True,
        )   
        bbox, w, h = self._calculate_text_bbox(
            self._rtext_list, self.rtext, ax    
        )
        rect = patches.Rectangle(
            (bbox.xmin, bbox.ymin-h), w, h, 
            linewidth=1, edgecolor='None', facecolor='None', alpha=0.25
        )
        ax.add_patch(rect)

    def _setup_bmisc(self):
        ax = self.ax_bmisc
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        pos = [xlims[0] + (xlims[1]-xlims[0])*0.01, 
               ylims[0] + (ylims[1]-ylims[0])*0.99]
        self.btext = ax.text(
            *pos, "", 
            horizontalalignment='left',
            verticalalignment='top',
            fontsize=self._btext_fontsize,
            usetex=True,
            animated=True,
        )
        bbox, w, h = self._calculate_text_bbox(
            self._btext_list, self.btext, ax
        )
        rect = patches.Rectangle(
            (bbox.xmin, bbox.ymin-h), w, h, 
            linewidth=1, edgecolor='None', facecolor='None', alpha=0.25
        )
        ax.add_patch(rect)
        
    ######################
    ##  Update Methods  ##
    ######################

    def _update_main(self, i):
        ax = self.ax_main
        self.maintext.set_text(f"t={self.get_t(i):.3f}")
        # Update heatmap
        phi = self.heatmap_phis[i]
        self.heatmap.set_array(phi.ravel())
        # Update scatterplot
        xy = self.get_xy(i)
        self.scat_main.set_data(xy)
        # Update minima
        if self.fixed_point_info is not None:
            fps, types, fpcolors = self.get_fixed_point_info(i)
            style_to_fps = {sty: [] for sty in self.fixed_point_styles}
            for i, fp in enumerate(fps):
                style_to_fps[(types[i], fpcolors[i])].append(fp)
            
            for scatter, sty in zip(self.scatter_mins, self.fixed_point_styles):
                matching_fps = np.array(style_to_fps[sty])
                if len(matching_fps) > 0:
                    scatter.set_data(matching_fps.T)
                else:
                    scatter.set_data([], [])

        elif self.minima is not None:
            mins = self.get_minima(i)
            self.scatter_mins.set_data(mins)
        # Update gradient vector field
        grads = self.get_grads_mesh(i)
        if grads is not None:
            u, v = grads
            norms = np.sqrt(u*u + v*v)
            u = u/norms
            v = v/norms
            self.gradient_field.set_UVC(u, v)
            pcm = plt.get_cmap(self._gradient_field_cmap)
            cnorm = colors.Normalize(vmin=0, vmax=np.max(norms))
            self.gradient_field.set_color(pcm(cnorm(norms).flatten()))

    def _update_surf(self, i):
        ax = self.ax_surf
        # Update the surface plot
        self.surface[0].remove()
        self.surface[0] = ax.plot_surface(
            self.heatmeshx, 
            self.heatmeshy, 
            self.heatmap_phis[i].reshape(self.heatmeshx.shape), 
            vmin=self.heatmap_phis.min(),
            vmax=self.heatmap_phis.max(),
            cmap=DEFAULT_CMAP, 
            alpha=self._surface_alpha,
            linewidth=0.1,
            animated=True,
        )
        # Update scatter plot
        # NOTE: Cannot currently implement 3d scatterplot due to mpl issue.
        # self.surface_scatter.set_data_3d(*self.get_xy(i), self.phis[i])
        # self.surface_scatter.set_3d_properties(self.phis[i])
        

    def _update_clst(self, i):
        ax = self.ax_clst
        t = self.get_t(i)
        if self.clst_index >= len(self.ts_saved):
            return 
        clst_t = self.ts_saved[self.clst_index]
        if t >= clst_t:
            xy_saved = self.get_xy_saved(self.clst_index)
            if self.show_sim_vs_obs:
                xy_observed = self.get_observed_data(self.clst_index)
                self.scat_clst1.set_data(xy_observed)
                if self.losses is not None:
                    loss = self.losses[self.clst_index]
                    self.clsttext.set_text(f"loss: {loss:.3g}")
            else:
                self.scat_clst1.set_data(self.scat_clst2.get_data())
            self.scat_clst2.set_data(xy_saved)
            self.clst_index += 1

    def _update_sigs(self, i):
        t = self.get_t(i)
        sigs = self.get_sigs(i)
        for j, sig in enumerate(sigs):
            self._signal_markers[j].set_data([[t], [sig]])

    def _update_tilt(self, i):
        t = self.get_t(i)
        params = self.get_ps(i)
        for j, param in enumerate(params):
            self._param_markers[j].set_data([[t], [param]])

    def _update_bifs(self, i):
        params = self.get_ps(i)
        p0 = params[self.p0idx]
        p1 = params[self.p1idx]
        self._bif_marker.set_data([[p0], [p1]])

    def _update_rmisc(self, i):
        s = self.get_rtext(i)
        self.rtext.set_text(s)

    def _update_bmisc(self, i):
        s = self.get_btext(i)
        self.btext.set_text(s)

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
    
    def get_observed_data(self, i):
        return self.observed_data[i].T
    
    def get_sigs_saved(self, i):
        return self.sigs_saved[i,:]
    
    def get_ps_saved(self, i):
        return self.ps_saved[i,:]
    
    def get_minima(self, i):
        return self.minima[i].T
    
    def get_fixed_point_info(self, i):
        return (self.fixed_point_info[k][i] for k in range(3))
    
    def get_rtext(self, i):
        return "" if i >= len(self._rtext_list) else self._rtext_list[i]
    
    def get_btext(self, i):
        return "" if i >= len(self._btext_list) else self._btext_list[i]
    
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

    def _plot_bifcurves(self, ax, linestyle='-', linewidth=1, color=None):
        ncurves = len(self.bifcurves)
        for i in range(ncurves):
            curve = self.bifcurves[i]
            if color is not None:
                c = color
            elif self.bifcolors is not None:
                c = self.bifcolors[i]
            else:
                c = self._bifcurvecolor
            ax.plot(
                curve[:,0], curve[:,1], 
                linewidth=linewidth,
                linestyle=linestyle,
                color=c, 
            )

    def _generate_rtext_list(self):
        txtlist = []
        for i in range(self.nframes):
            txtlist.append(self._build_rtext(i))
        return txtlist

    def _generate_btext_list(self):
        txtlist = []
        for i in range(self.nframes):
            txtlist.append(self._build_btext(i))
        return txtlist
    
    def _build_rtext(self, i):
        """Right miscellaneous box text."""
        t = self.get_t(i)
        sig = self.get_sigs(i)
        sig_str = ", ".join([f"{x:.4f}" for x in sig])
        tau = self.get_ps(i)
        tau_str = ", ".join([f"{x:.4f}" for x in tau])
        dt0 = self.model.get_dt0()
        sigma = self.model.get_sigma()
        ncells = self.model.get_ncells()
        s = f"$t={t:.3f}$" 
        s += f"\n$\\boldsymbol{{s}}=({sig_str})^T$"
        s += f"\n$\\boldsymbol{{\\tau}}=({tau_str})^T$\n"
        s += f"\nModel $\sigma: {sigma:.3g}$"
        s += f"\nModel $\\mathtt{{dt0}}: {dt0:.3g}$"
        s += f"\nModel $\\mathtt{{ncells}}: {ncells}$"
        if self.sigparams_str:
            s += "\n\n" + self.sigparams_str
        return s
    
    def _build_btext(self, i):
        """Bottom miscellaneous box text."""
        s = self.model_info_string + f"\\newline\\texttt{{{self.note_string}}}"
        return s
    
    def _calculate_text_bbox(self, string_list, text, ax):
        transf = ax.transData.inverted()
        maxw = 0
        maxh = 0
        s0 = text.get_text()
        bbox = text.get_window_extent().transformed(transf)
        for s in string_list:
            text.set_text(s)
            bb = text.get_window_extent().transformed(transf)
            w, h = bb.width, bb.height
            if w > maxw:
                maxw = w
            if h > maxh:
                maxh = h
        text.set_text(s0)
        return bbox, maxw, maxh

