"""Figure 1 Script (Landscape Models)

Generate plots used in Figure 1 of the accompanying manuscript.
"""

import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
plt.style.use('figures/manuscript/styles/fig_standard.mplstyle')

from plnn.pl import plot_landscape
from plnn.helpers import get_phi1_fixed_points, get_phi2_fixed_points
from plnn.data_generation.signals import get_sigmoid_function
from plnn.models.algebraic_pl import AlgebraicPL


OUTDIR = "figures/manuscript/out/fig1_landscape_models"
SAVEPLOTS = True

SEED = 1234125123


os.makedirs(OUTDIR, exist_ok=True)

def func_phi1_star(x, y, p1=0, p2=0):
    return x**4 + y**4 + y**3 - 4*x*x*y + y*y + p1*x + p2*y

def func_phi2_star(x, y, p1=0, p2=0):
    return x**4 + y**4 + x**3 - 2*x*y*y - x*x + p1*x + p2*y

sf = 1/2.54  # scale factor from [cm] to inches

FP_MARKERS = {
    'saddle': 'x',
    'minimum': 'o',
    'maximum': '^',
}

ANNOTATION_FONTSIZE = 8
PARAM_MARKERSIZE = 4

##############################################################################
##############################################################################
##  Untilted binary choice landscape.

FIGNAME1 = "phi1_heatmap_untilted"
FIGSIZE1 = (5*sf, 5*sf)
FIGNAME2 = "phi1_landscape_untilted"
FIGSIZE2 = (6*sf, 6*sf)

r = 3       # box radius
res = 50   # resolution
lognormalize = True
clip = None

fps, fp_types, fp_colors = get_phi1_fixed_points([[0, 0]])

ax = plot_landscape(
    func_phi1_star, r=r, res=res, params=[0, 0], 
    lognormalize=lognormalize,
    clip=clip,
    title=f"$\\boldsymbol{{\\tau}}=\langle 0, 0\\rangle$",
    ncontours=10,
    contour_linewidth=0.5,
    contour_linealpha=0.5,
    include_cbar=True,
    cbar_title="$\ln\phi$" if lognormalize else "$\phi$",
    equal_axes=True,
    figsize=FIGSIZE1,
    show=True
);
for fp, fp_type, fp_color in zip(fps[0], fp_types[0], fp_colors[0]):
    marker = FP_MARKERS[fp_type]
    ax.plot(
        fp[0], fp[1],
        color=fp_color, 
        marker=marker,
        markersize=3,
        markeredgecolor='w',
        markeredgewidth=0.2 if marker == 'o' else 0.5,
        
    )
if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/{FIGNAME1}")
plt.close()

ax = plot_landscape(
    func_phi1_star, r=r, res=res, params=[0, 0], 
    plot3d=True,
    lognormalize=False,
    normalize=True,
    minimum=50,
    clip=100,
    include_cbar=False,
    title=f"",
    cbar_title="$\ln\phi$",
    alpha=0.75,
    xlims=[-3.5, 3.5],
    ylims=[-3.5, 3.5],
    zlims=[0, 150],
    zlabel="$\phi$",
    view_init=[35, -45],  # elevation, aximuthal 
    ncontours=10,
    contour_linewidth=0.5,
    contour_linealpha=0.5,
    equal_axes=True,
    tight_layout=True,
    figsize=FIGSIZE2,
    show=True,
);
for fp, fp_type, fp_color in zip(fps[0], fp_types[0], fp_colors[0]):
    marker = FP_MARKERS[fp_type]
    ax.plot(
        fp[0], fp[1], 0,
        color=fp_color, 
        marker=marker,
        markersize=3,
        markeredgecolor='w',
        markeredgewidth=0.2 if marker == 'o' else 0.5,
    )
# ax.tick_params(axis='x', which='both', pad=-5)
# ax.tick_params(axis='y', which='both', pad=-5)
# ax.tick_params(axis='z', which='both', pad=0)
ax.xaxis.labelpad = -15
ax.yaxis.labelpad = -15
ax.zaxis.labelpad = -15
ax.grid(True)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
# ax.set_xlabel("")
# ax.set_ylabel("")
# ax.set_zlabel("")
if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/{FIGNAME2}", bbox_inches='tight')
plt.close()

##############################################################################
##############################################################################
##  Untilted binary flip landscape.

FIGNAME1 = "phi2_heatmap_untilted"
FIGSIZE1 = (5*sf, 5*sf)
FIGNAME2 = "phi2_landscape_untilted"
FIGSIZE2 = (6*sf, 6*sf)

r = 3       # box radius
res = 50   # resolution
lognormalize = True
clip = None

fps, fp_types, fp_colors = get_phi2_fixed_points([[0, 0]])

ax = plot_landscape(
    func_phi2_star, r=r, res=res, params=[0, 0], 
    lognormalize=lognormalize,
    clip=clip,
    title=f"$\\boldsymbol{{\\tau}}=\langle 0, 0\\rangle$",
    ncontours=10,
    contour_linewidth=0.5,
    include_cbar=True,
    cbar_title="$\ln\phi$" if lognormalize else "$\phi$",
    equal_axes=True,
    figsize=FIGSIZE1,
    show=True,
);
for fp, fp_type, fp_color in zip(fps[0], fp_types[0], fp_colors[0]):
    marker = FP_MARKERS[fp_type]
    ax.plot(
        fp[0], fp[1],
        color=fp_color, 
        marker=marker,
        markersize=3,
        markeredgecolor='w',
        markeredgewidth=0.2 if marker == 'o' else 0.5,
    )
if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/{FIGNAME1}")
plt.close()

ax = plot_landscape(
    func_phi2_star, r=r, res=res, params=[0, 0], 
    plot3d=True,
    lognormalize=False,
    normalize=False,
    minimum=50,
    clip=100,
    title=f"$\phi_{{bf}}$",
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
    figsize=FIGSIZE2,
    show=True,
);
for fp, fp_type, fp_color in zip(fps[0], fp_types[0], fp_colors[0]):
    marker = FP_MARKERS[fp_type]
    ax.plot(
        fp[0], fp[1],
        color=fp_color, 
        marker=marker,
        markersize=3,
        markeredgecolor='w',
        markeredgewidth=0.2 if marker == 'o' else 0.5,
    )
if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/{FIGNAME2}")
plt.close()

##############################################################################
##############################################################################
##  Tilted binary choice landscape

FIGNAME = "phi1_heatmap_tilted"  # appended with index i
FIGSIZE = (4*sf, 4*sf)

PARAMS_PHI1 = [
    ([ 0,  0.5], 0),
    ([-1.00,  1.00], 1),
    ([ 1.00,  1.00], 2),
    ([-0.50,  0.00], 3),
    ([ 0.50,  0.00], 4),
]
r = 3       # box radius
res = 50   # resolution
lognormalize = True
clip = None

for i, (p, label_number) in enumerate(PARAMS_PHI1):
    title = f"$\\boldsymbol{{\\tau}}=" + \
                f"\langle{p[0]:.2g},{p[1]:.2g}\\rangle$"
    ax = plot_landscape(
        func_phi1_star, r=r, res=res, params=p, 
        lognormalize=lognormalize,
        clip=clip,
        title=title,
        cbar_title="$\ln\phi$",
        include_cbar=False,
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
    ax.text(
        0.99, 0, str(label_number), 
        color='k',
        fontsize=ANNOTATION_FONTSIZE, 
        ha='right', 
        va='bottom', 
        transform=ax.transAxes
    )
    fps, fp_types, fp_colors = get_phi1_fixed_points([p])
    for fp, fp_type, fp_color in zip(fps[0], fp_types[0], fp_colors[0]):
        marker = FP_MARKERS[fp_type]
        ax.plot(
            fp[0], fp[1],
            color=fp_color, 
            marker=marker,
            markersize=3,
            markeredgecolor='w',
            markeredgewidth=0.2 if marker == 'o' else 0.5,
        )
    if SAVEPLOTS:
        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/{FIGNAME}_{i}", bbox_inches='tight')
    plt.close()


##############################################################################
##############################################################################
##  Tilt time course for binary choice
FIGNAME = "phi1_tau_timecourse"  # appended with index i
FIGSIZE = (6*sf, 4*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

sigparams = np.array([
    [2.0, 0.0, 0.5, 2.0],
    [6.0, 0.5, 0.0, 4.0],
])

ts = np.linspace(0, 8, 1001, endpoint=True)
sigfunc = get_sigmoid_function(*sigparams.T)
taus_timecourse = sigfunc(ts)

key = jrandom.PRNGKey(seed=SEED)

key, subkey = jrandom.split(key, 2)
model, hyperparams = AlgebraicPL.make_model(
    key=subkey,
    dtype=jnp.float64,
    algebraic_phi_id="phi1",
    tilt_weights=[[1, 0],[0, 1]],
    tilt_bias=[0, 0],
    sigma_init=0.2,
    signal_type="sigmoid",
    nsigparams=4,
    dt0=0.05,
)

ncells = 10
x0_val = (0.0, -0.5)
tfin = 8
burnin = 1
dt_save = 0.5

# Initial condition
key, subkey = jrandom.split(key, 2)
x0 = np.zeros([ncells, 2])
x0[:] = x0_val

# Simulate particles in the landscape
key, subkey = jrandom.split(key, 2)
ts_saved, xs_saved, sigs_saved, ps_saved = model.run_landscape_simulation(
    x0, tfin, dt_save, sigparams, subkey, 
    burnin=burnin
)

ts_saved = ts_saved[0]
xs_saved = xs_saved[0]
sigs_saved = sigs_saved[0]
ps_saved = ps_saved[0]

plot_ts = [0, 2, 4, 6, 8]

ax.plot(
    ts, taus_timecourse, 
    label=["$\\tau_1$", "$\\tau_2$"])
ax.legend(loc='lower center')

for plot_t in plot_ts:
    ax.axvline(plot_t, color='k', alpha=0.5, linestyle='--')

if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/{FIGNAME}", transparent=True)
plt.close()


##############################################################################
##############################################################################
##  Landscape heatmaps along tau timecourse
FIGNAME = "phi1_timecourse_heatmaps"  # appended with index i
FIGSIZE = (2.5*sf, 2.5*sf)

res = 50
r = 2.5

for i, plot_t in enumerate(plot_ts):

    idx_in_saved_results = np.where(ts_saved == plot_t)[0][0]
    x_state = xs_saved[idx_in_saved_results]

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    p = sigfunc(plot_t).flatten()
    plot_landscape(
        func_phi1_star, r=r, res=res, params=p,
        lognormalize=lognormalize,
        clip=clip,
        title="",
        include_cbar=False,
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
        ax=ax,
    )

    fps, fp_types, fp_colors = get_phi1_fixed_points([np.round(p, 6)])
    for fp, fp_type, fp_color in zip(fps[0], fp_types[0], fp_colors[0]):
        marker = FP_MARKERS[fp_type]
        ax.plot(
            fp[0], fp[1],
            color=fp_color, 
            marker=marker,
            markersize=3,
            markeredgecolor='w',
            markeredgewidth=0.2 if marker == 'o' else 0.5,
        )
    
    ax.plot(
        x_state[:,0], x_state[:,1], '.',
        color='cyan',
        markersize=1,
        alpha=0.8,
    )

    if SAVEPLOTS:
        plt.savefig(f"{OUTDIR}/{FIGNAME}_{i}", transparent=True)
    plt.close()

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
r = 3       # box radius
res = 50   # resolution
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
        cbar_title="$\ln\phi$",
        ncontours=10,
        contour_linewidth=0.5,
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
        marker = FP_MARKERS[fp_type]
        ax.plot(
            fp[0], fp[1],
            color=fp_color, 
            marker=marker,
            markersize=3,
            markeredgecolor='w',
            markeredgewidth=0.2 if marker == 'o' else 0.5,
        )
    if SAVEPLOTS:
        plt.savefig(f"{OUTDIR}/{FIGNAME}_{i}")
    plt.close()

##############################################################################
##############################################################################
##  Bifurcation Diagram for Binary Choice (phi1)

from cont.binary_choice import plot_binary_choice_bifurcation_diagram

FIGSIZE = (5*sf, 5*sf)

fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)

plot_binary_choice_bifurcation_diagram(
    ax=ax,
    xlabel="$\\tau_1$",
    ylabel="$\\tau_2$",
)

ax.plot(
    taus_timecourse[:,0], taus_timecourse[:,1],
    color='grey', 
    linewidth=1.5,
    alpha=0.8,
    linestyle='-',
)

for p, label_number in PARAMS_PHI1:
    ax.plot(*p, '.k', alpha=1.0, markersize=PARAM_MARKERSIZE)
    ax.text(*p, label_number, fontsize=ANNOTATION_FONTSIZE)

if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/phi1_bifdiagram", bbox_inches='tight')

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
    ax.plot(*p, '.k', alpha=1.0, markersize=PARAM_MARKERSIZE)

if SAVEPLOTS:
    plt.savefig(f"{OUTDIR}/phi2_bifdiagram")

plt.close()

##############################################################################
##############################################################################
