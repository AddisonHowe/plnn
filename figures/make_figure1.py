"""Figure 1 Script

Generate plots used in Figure 1 of the accompanying manuscript.
"""

import os
import matplotlib.pyplot as plt
plt.style.use('figures/fig1.mplstyle')

from plnn.pl import plot_landscape


OUTDIR = "figures/out/fig1_out"
SAVEPLOTS = True

os.makedirs(OUTDIR, exist_ok=True)

def func_phi1_star(x, y, p1=0, p2=0):
    return x**4 + y**4 + y**3 - 4*x*x*y + y*y + p1*x + p2*y

def func_phi2_star(x, y, p1=0, p2=0):
    return x**4 + y**4 + x**3 - 2*x*y*y - x*x + p1*x + p2*y

sf = 1/2.54  # scale factor from [cm] to inches

##############################################################################
##############################################################################
##  Untilted binary choice landscape.

FIGNAME1 = "phi1_heatmap_untilted"
FIGSIZE1 = (5*sf, 5*sf)
FIGNAME2 = "phi1_landscape_untilted"
FIGSIZE2 = (5*sf, 5*sf)

r = 4       # box radius
res = 200   # resolution
lognormalize = True
clip = None

plot_landscape(
    func_phi1_star, r=r, res=res, params=[0, 0], 
    lognormalize=lognormalize,
    clip=clip,
    title=f"$\phi$ untilted",
    title_fontsize=8,
    ncontours=10,
    contour_linewidth=0.5,
    include_cbar=True,
    cbar_title="$\log\phi$" if lognormalize else "$\phi$",
    equal_axes=True,
    saveas=f"{OUTDIR}/{FIGNAME1}" if SAVEPLOTS else None,
    figsize=FIGSIZE1,
);

plot_landscape(
    func_phi1_star, r=r, res=res, params=[0, 0], 
    plot3d=True,
    lognormalize=False,
    normalize=True,
    minimum=50,
    clip=100,
    include_cbar=False,
    title=f"$\phi$ untilted",
    title_fontsize=9,
    alpha=0.75,
    xlims=[-3.5, 3.5],
    ylims=[-3.5, 3.5],
    zlims=[0, 150],
    zlabel="$\phi$",
    view_init=[35, -45],  # elevation, aximuthal 
    ncontours=10,
    contour_linewidth=0.5,
    equal_axes=True,
    tight_layout=True,
    saveas=f"{OUTDIR}/{FIGNAME2}" if SAVEPLOTS else None,
    figsize=FIGSIZE2,
);

##############################################################################
##############################################################################
##  Untilted binary flip landscape.

FIGNAME1 = "phi2_heatmap_untilted"
FIGSIZE1 = (5*sf, 5*sf)
FIGNAME2 = "phi2_landscape_untilted"
FIGSIZE2 = (6*sf, 6*sf)

r = 4       # box radius
res = 200   # resolution
lognormalize = True
clip = None

plot_landscape(
    func_phi2_star, r=r, res=res, params=[0, 0], 
    lognormalize=lognormalize,
    clip=clip,
    title=f"$\phi_2^*$ (untilted)",
    title_fontsize=8,
    ncontours=10,
    contour_linewidth=0.5,
    include_cbar=True,
    cbar_title="$\log\phi$" if lognormalize else "$\phi$",
    equal_axes=True,
    saveas=f"{OUTDIR}/{FIGNAME1}" if SAVEPLOTS else None,
    figsize=FIGSIZE1,
);

plot_landscape(
    func_phi2_star, r=r, res=res, params=[0, 0], 
    plot3d=True,
    lognormalize=False,
    normalize=False,
    minimum=50,
    clip=100,
    title=f"$\phi_2^*$ (untilted)",
    title_fontsize=9,
    include_cbar=False,
    cbar_title="$\phi$",
    alpha=0.75,
    equal_axes=True,
    xlims=[-3.5, 3.5],
    ylims=[-3.5, 3.5],
    zlims=[0, 150],
    view_init=[35, -80],  # elevation, aximuthal 
    ncontours=10,
    contour_linewidth=0.5,
    saveas=f"{OUTDIR}/{FIGNAME2}" if SAVEPLOTS else None,
    figsize=FIGSIZE2,
);

##############################################################################
##############################################################################
##  Tilted binary choice landscape

FIGNAME = "phi1_heatmap_tilted"  # appended with index i
FIGSIZE = (4*sf, 4*sf)

PARAMS_PHI1 = [
    [ 0,  0],
    [-0.25,  1.00],
    [ 0.25,  1.00],
    [-0.50,  0.00],
    [ 0.50,  0.00],
]
r = 4       # box radius
res = 200   # resolution
lognormalize = True
clip = None

for i, signal in enumerate(PARAMS_PHI1):
    title = f"$\\boldsymbol{{\\tau}}=" + \
                f"\langle{-signal[0]:.2g},{signal[1]:.2g}\\rangle$"
    plot_landscape(
        func_phi1_star, r=r, res=res, params=signal, 
        lognormalize=lognormalize,
        clip=clip,
        title=title,
        cbar_title="$\log\phi$",
        include_cbar=False,
        ncontours=10,
        contour_linewidth=0.5,
        xlabel=None,
        ylabel=None,
        xticks=False,
        yticks=False,
        equal_axes=True,
        saveas=f"{OUTDIR}/{FIGNAME}_{i}" if SAVEPLOTS else None,
        figsize=FIGSIZE,
    );

##############################################################################
##############################################################################
##  Tilted binary flip landscape

FIGNAME = "phi2_heatmap_tilted"  # appended with index i
FIGSIZE = (4*sf, 4*sf)

PARAMS_PHI2 = [
    [ 1.0,  1.0],
    [ 1.0, -1.0],
    [-1.0,  1.0],
    [-1.0, -1.0],
]
r = 4       # box radius
res = 200   # resolution
lognormalize = True
clip = None

for i, signal in enumerate(PARAMS_PHI2):
    title = f"$\\boldsymbol{{\\tau}}=" + \
                f"\langle{signal[0]:.2g},{signal[1]:.2g}\\rangle$"
    plot_landscape(
        func_phi2_star, r=r, res=res, params=signal, 
        lognormalize=lognormalize,
        clip=clip,
        title=title,
        include_cbar=False,
        cbar_title="$\log\phi$",
        ncontours=10,
        contour_linewidth=0.5,
        xlabel=None,
        ylabel=None,
        xticks=False,
        yticks=False,
        equal_axes=True,
        saveas=f"{OUTDIR}/{FIGNAME}_{i}" if SAVEPLOTS else None,
        figsize=FIGSIZE,
    );

##############################################################################
##############################################################################
##  Bifurcation Diagram for Binary Choice (phi1)

from cont.binary_choice import plot_binary_choice_bifurcation_diagram

FIGSIZE = (4*sf, 4*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

plot_binary_choice_bifurcation_diagram(
    ax=ax,
    xlabel="$\\tau_1$",
    ylabel="$\\tau_2$",
)

for p in PARAMS_PHI1:
    ax.plot(*p, '.k', alpha=0.8, markersize=1)

if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/phi1_bifdiagram")

plt.close()

##############################################################################
##############################################################################
##  Bifurcation Diagram for Binary Flip (phi2)

from cont.binary_flip import plot_binary_flip_bifurcation_diagram

FIGSIZE = (4*sf, 4*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

plot_binary_flip_bifurcation_diagram(
    ax=ax,
)

for p in PARAMS_PHI2:
    ax.plot(*p, '.k', alpha=0.8)

if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/phi2_bifdiagram")

plt.close()

##############################################################################
##############################################################################
