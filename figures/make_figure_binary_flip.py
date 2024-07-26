"""Figure binary-flip

"""

import os
import matplotlib.pyplot as plt
plt.style.use('figures/styles/fig1.mplstyle')

from plnn.pl import plot_landscape
from plnn.helpers import get_phi2_fixed_points


OUTDIR = "figures/out/fig_binary_flip"
SAVEPLOTS = True

os.makedirs(OUTDIR, exist_ok=True)

def func_phi2_star(x, y, p1=0, p2=0):
    return x**4 + y**4 + x**3 - 2*x*y*y - x*x + p1*x + p2*y

sf = 1/2.54  # scale factor from [cm] to inches

FP_MARKERS = {
    'saddle': 'x',
    'minimum': '*',
    'maximum': 'o',
}

##############################################################################
##############################################################################
##  Untilted binary flip landscape.

FIGNAME1 = "phi2_heatmap_untilted"
FIGSIZE1 = (5*sf, 5*sf)
FIGNAME2 = "phi2_landscape_untilted"
FIGSIZE2 = (6*sf, 6*sf)

r = 4       # box radius
res = 100   # resolution
lognormalize = True
clip = None

fps, fp_types, fp_colors = get_phi2_fixed_points([[0, 0]])

ax = plot_landscape(
    func_phi2_star, r=r, res=res, params=[0, 0], 
    lognormalize=lognormalize,
    clip=clip,
    title=f"$\\boldsymbol{{\\tau}}=\langle 0, 0\\rangle$",
    title_fontsize=8,
    ncontours=10,
    contour_linewidth=0.5,
    contour_linealpha=0.5,
    include_cbar=True,
    cbar_title="$\log\phi$" if lognormalize else "$\phi$",
    equal_axes=True,
    figsize=FIGSIZE1,
    show=True,
);
for fp, fp_type, fp_color in zip(fps[0], fp_types[0], fp_colors[0]):
    ax.plot(
        fp[0], fp[1],
        color=fp_color, 
        marker=FP_MARKERS[fp_type],
        markersize=2,
    )
if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/{FIGNAME1}", bbox_inches='tight')
plt.close()

ax = plot_landscape(
    func_phi2_star, r=r, res=res, params=[0, 0], 
    plot3d=True,
    lognormalize=False,
    normalize=False,
    minimum=50,
    clip=100,
    include_cbar=False,
    title=f"",
    title_fontsize=9,
    cbar_title="$\ln\phi$",
    alpha=0.75,
    xlims=[-3.5, 3.5],
    ylims=[-3.5, 3.5],
    zlims=[0, 150],
    zlabel="$\phi$",
    view_init=[35, -80],  # elevation, aximuthal 
    ncontours=10,
    contour_linewidth=0.5,
    contour_linealpha=0.5,
    equal_axes=True,
    # tight_layout=True,
    figsize=FIGSIZE2,
    show=True,
);
for fp, fp_type, fp_color in zip(fps[0], fp_types[0], fp_colors[0]):
    ax.plot(
        fp[0], fp[1], 0,
        color=fp_color, 
        marker=FP_MARKERS[fp_type],
        markersize=2,
    )
ax.tick_params(axis='x', which='both', pad=-5)
ax.tick_params(axis='y', which='both', pad=0)
ax.tick_params(axis='z', which='both', pad=0)
ax.xaxis.labelpad = -10
ax.yaxis.labelpad = 0
ax.zaxis.labelpad = 0
if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/{FIGNAME2}")
plt.close()

##############################################################################
##############################################################################
##  Tilted binary flip landscape

FIGNAME = "phi2_heatmap_tilted"  # appended with index i
FIGSIZE = (4*sf, 4*sf)

PARAMS_PHI2 = [
    [ 0,    0],
    [-0.75, 0.25],
    [-0.75, -0.25],
    [-1.0,  1.0],
    [-1.0, -1.0],
    [-1.5,  0.5],
    [-1.5, -0.5],
]
r = 4       # box radius
res = 100   # resolution
lognormalize = True
clip = None

for i, p in enumerate(PARAMS_PHI2):
    title = f"$\\boldsymbol{{\\tau}}=" + \
                f"\langle{p[0]:.2g},{p[1]:.2g}\\rangle$"
    ax = plot_landscape(
        func_phi2_star, r=r, res=res, params=p, 
        lognormalize=lognormalize,
        clip=clip,
        title=title,
        include_cbar=False,
        cbar_title="$\log\phi$",
        ncontours=10,
        contour_linewidth=0.5,
        contour_linealpha=0.5,
        xlabel=None,
        ylabel=None,
        xticks=False,
        yticks=False,
        equal_axes=True,
        figsize=FIGSIZE,
        show=True,
    );
    fps, fp_types, fp_colors = get_phi2_fixed_points([p])
    for fp, fp_type, fp_color in zip(fps[0], fp_types[0], fp_colors[0]):
        ax.plot(
            fp[0], fp[1],
            color=fp_color, 
            marker=FP_MARKERS[fp_type],
            markersize=2,
        )
    if SAVEPLOTS:
        plt.savefig(f"{OUTDIR}/{FIGNAME}_{i}", bbox_inches='tight')
    plt.close()

##############################################################################
##############################################################################
##  Bifurcation Diagram for Binary Flip (phi2)

from cont.binary_flip import plot_binary_flip_bifurcation_diagram

FIGSIZE = (4*sf, 4*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

plot_binary_flip_bifurcation_diagram(
    ax=ax,
    xlabel="$\\tau_1$",
    ylabel="$\\tau_2$",
)

for p in PARAMS_PHI2:
    ax.plot(*p, '.k', alpha=0.8, markersize=3)

if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/phi2_bifdiagram", bbox_inches='tight')

plt.close()

##############################################################################
##############################################################################
