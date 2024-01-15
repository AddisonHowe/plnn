import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PIL import Image
import time
import matplotlib.animation as animation

"""

"""

class SimulationAnimator:
    """
    """

    savegif = True
    fps = 2                 # default frames per second
    dpi = 100               # default resolution
    interval = 200          # default delay between frames in milliseconds.
    figsize = (12, 9)    # default figure size

    main_scatter_color = 'k'
    axlimbuffer = 0.05  # percent buffer on axis lims

    def __init__(self, ts, xys, **kwargs):
        """
        ts         : (nsaves,)
        xys        : (nsaves, ncells, ndim)
        ps (opt)   : (nsaves, [param_shape])
        """
        #~~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~#
        ps = kwargs.get('ps', None)
        xlims = kwargs.get('xlims', None)
        ylims = kwargs.get('ylims', None)
        grads = kwargs.get('grads', None)
        grad_func = kwargs.get('grad_func', None)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.ts = ts
        self.xys = xys
        self.xs = xys[:,:,0]
        self.ys = xys[:,:,1]
        self.ps = ps
        self.grads = grads
        self.grad_func = grad_func
        self.nframes = len(self.ts)
        self.xlims = xlims if xlims else self.buffer_lims([np.min(self.xs), 
                                                           np.max(self.xs)])
        self.ylims = ylims if ylims else self.buffer_lims([np.min(self.ys), 
                                                           np.max(self.ys)])

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
            fps = int(self.nframes / duration)
        
        self.fig = plt.figure(dpi=self.dpi, figsize=figsize, 
                              constrained_layout=True,)
        
        self.ax_main = plt.subplot2grid((4, 5), (0, 0), colspan=2, rowspan=2)
        self.ax_clst = plt.subplot2grid((4, 5), (2, 0), colspan=2, rowspan=2)
        self.ax_rsig = plt.subplot2grid((4, 5), (0, 2), colspan=2, rowspan=1)
        self.ax_esig = plt.subplot2grid((4, 5), (1, 2), colspan=2, rowspan=1)
        self.ax_dist = plt.subplot2grid((4, 5), (2, 2), colspan=2, rowspan=2)
        self.ax_text = plt.subplot2grid((4, 5), (0, 4), colspan=1, rowspan=4)

        print("Generating movie...")
        sys.stdout.flush()
        tic0 =time.time()
        self.ani = animation.FuncAnimation(
            self.fig, self.update, 
            frames=self.nframes, 
            interval=interval, 
			init_func=self.setup, 
            blit=True
        )
        tic1 = time.time() 
        print(f"Finished in {tic1-tic0:.3g} seconds.")
        sys.stdout.flush()
        if savepath:
            print(f"Saving animation to {savepath}")
            sys.stdout.flush()
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

    def setup(self):
        """Initialize each axis and text."""
        self._setup_main()
        self._setup_clst()
        self._setup_rsig()
        self._setup_esig()
        self._setup_dist()
        self._setup_text()
        return self.scat_main,# self.esig0_scat, self.esig1_scat,
        
    def update(self, i):
        """Update each axis and text."""
        self._update_main(i)
        self._update_clst(i)
        self._update_rsig(i)
        self._update_esig(i)
        self._update_dist(i)
        self._update_text(i)
        return self.scat_main,# self.esig0_scat, self.esig1_scat,

    #####################
    ##  Setup Methods  ##
    #####################
        
    def _setup_main(self):
        ax = self.ax_main
        # Initialize main axis
        self.scat_main, = ax.plot(
            [], [], marker='.', markersize=4, 
            color=self.main_scatter_color, 
            alpha=0.75, linestyle='None', 
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
            self.meshx, self.meshy, 
            np.zeros(self.meshx.shape), np.zeros(self.meshy.shape),
            color=self.main_scatter_color, 
            alpha=0.5, 
            animated=True
        )

    def _setup_clst(self):
        ax = self.ax_clst
        ax.set_xlabel(f"$x$")
        ax.set_ylabel(f"$y$")
        ax.axis([*self.xlims, *self.ylims])
        # Text
        pos = [self.xlims[0] + (self.xlims[1]-self.xlims[0])*.5, 
               self.ylims[0] + (self.ylims[1]-self.ylims[0])*.95]
        self.clsttext = ax.text(*pos, "", fontsize='small')

    def _setup_rsig(self):
        pass

    def _setup_esig(self):
        pass

    def _setup_dist(self):
        pass

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
        self.sim_param_str = "Parameters:\n..."

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

    def _update_clst(self, i):
        pass

    def _update_rsig(self, i):
        pass

    def _update_esig(self, i):
        pass

    def _update_dist(self, i):
        pass

    def _update_text(self, i):
        t = self.get_t(i)
        s = f"$t={t:.3f}$\n" + self.sim_param_str
        self.text.set_text(s)

    ######################
    ##  Helper Methods  ##
    ######################

    def get_t(self, i):
        return self.ts[i]
    
    def get_xy(self, i):
        return self.xys[i, :].T
    
    def get_ps(self, i):
        if self.ps is None:
            return None
        return self.ps[i,:]
    
    def buffer_lims(self, lims):
        buffer = self.axlimbuffer * (lims[1] - lims[0])
        return [lims[0] - buffer, lims[1] + buffer]
        
    def get_grads_mesh(self, i):
        if self.grads is not None:
            return self.grads[i]
        elif self.grad_func:
            x = self.meshx.flatten()
            y = self.meshy.flatten()
            p = self.get_ps(i)
            gradx, grady = self.grad_func(x, y, p)
            return gradx, grady
        else:
            return None
