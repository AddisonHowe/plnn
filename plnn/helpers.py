"""General Helper Functions

"""

import numpy as np
import sympy
from plnn.pl.config import PHI1_COLOR_DICT, PHI2_COLOR_DICT

def get_phi1_fixed_points(ps):
    """
    """
    x, y, p1, p2 = sympy.symbols('x y p1 p2')
    eq1 = 4*x**3 - 8*x*y + p1
    eq2 = 4*y**3 + 3*y**2 - 4*x*x + 2*y + p2
    y_eq = (4*x**3 + p1) / (8*x)  # y in terms of x, assuming x nonzero

    all_fixed_points = []
    all_types = []
    all_colors = []
    for i, p in enumerate(ps):
        fps = []
        types = []
        cols = []
        
        p1val, p2val = p
        eq2_sub = eq2.subs([(y, y_eq), (p1, p1val), (p2, p2val)])
        xsolset = sympy.solveset(sympy.Eq(eq2_sub, 0), x)
        
        for xsol in xsolset:
            ysol = y_eq.subs([(x, xsol), (p1, p1val)])
            if xsol.is_real and ysol.is_real:
                fp = (xsol.evalf(), ysol.evalf())
                fps.append(fp)
                cols.append(get_phi1_fixed_point_color(fp))
                types.append(get_phi1_fixed_point_type(fp))
        # Check case p1=0 implies x=0
        if p1val == 0:  
            xsol = 0.
            eq2_sub = eq2.subs([(x, 0.), (p1, p1val), (p2, p2val)])
            ysolset = sympy.solveset(sympy.Eq(eq2_sub, 0), y)
            for ysol in ysolset:
                if ysol.is_real:
                    fp = (xsol, ysol.evalf())
                    fps.append(fp)
                    cols.append(get_phi1_fixed_point_color(fp))
                    types.append(get_phi1_fixed_point_type(fp))

        all_fixed_points.append(fps)
        all_types.append(types)
        all_colors.append(cols)
    
    return all_fixed_points, all_types, all_colors


def get_phi1_fixed_point_color(fp):
    x, y = fp
    is_saddle = (3*x*x - 2*y) * (6*y*y+3*y+1) - 8*x*x <= 0
    if is_saddle:
        return PHI1_COLOR_DICT['saddle']
    if y <= 0:
        return PHI1_COLOR_DICT['c']
    if x > 0:
        return PHI1_COLOR_DICT['r']
    return PHI1_COLOR_DICT['l']


def get_phi1_fixed_point_type(fp):
    x, y = fp
    is_saddle = (3*x*x - 2*y) * (6*y*y+3*y+1) - 8*x*x <= 0
    return 'saddle' if is_saddle else 'minimum'


def get_phi2_fixed_points(ps):
    """
    """
    x, y, p1, p2 = sympy.symbols('x y p1 p2')
    eq1 = 4*x**3 + 3*x**2 - 2*y**2 -2*x + p1
    eq2 = 4*y**3 - 4*x*y + p2
    x_eq = (4*y**3 + p2) / (4*y)  # x in terms of y, assuming y nonzero

    all_fixed_points = []
    all_types = []
    all_colors = []
    for i, p in enumerate(ps):
        fps = []
        types = []
        cols = []
        
        p1val, p2val = p
        eq1_sub = eq1.subs([(x, x_eq), (p1, p1val), (p2, p2val)])
        ysolset = sympy.solveset(sympy.Eq(eq1_sub, 0), y)
        
        for ysol in ysolset:
            xsol = x_eq.subs([(y, ysol), (p2, p2val)])
            if xsol.is_real and ysol.is_real:
                fp = (xsol.evalf(), ysol.evalf())
                fps.append(fp)
                cols.append(get_phi2_fixed_point_color(fp))
                types.append(get_phi2_fixed_point_type(fp))
        # Check case p2=0 implies y=0 is a solution
        if p2val == 0:  
            ysol = 0.
            eq1_sub = eq1.subs([(y, 0.), (p1, p1val), (p2, p2val)])
            xsolset = sympy.solveset(sympy.Eq(eq1_sub, 0), x)
            for xsol in xsolset:
                if xsol.is_real:
                    fp = (xsol.evalf(), ysol)
                    fps.append(fp)
                    cols.append(get_phi2_fixed_point_color(fp))
                    types.append(get_phi2_fixed_point_type(fp))

        all_fixed_points.append(fps)
        all_types.append(types)
        all_colors.append(cols)
    
    return all_fixed_points, all_types, all_colors


def get_phi2_fixed_point_color(fp):
    x, y = fp
    is_saddle = (12*x*x + 6*x - 2) * (12*y*y - 4*x) - 16*y*y <= 0
    if is_saddle:
        return PHI2_COLOR_DICT['saddle']
    if x < 0:
        return PHI2_COLOR_DICT['c']
    elif x < 0.2287:
        return PHI2_COLOR_DICT['maximum']
    elif y > 0:
        return PHI2_COLOR_DICT['t']
    else:
        return PHI2_COLOR_DICT['b']


def get_phi2_fixed_point_type(fp):
    x, y = fp
    is_saddle = (12*x*x + 6*x - 2) * (12*y*y - 4*x) - 16*y*y <= 0
    if is_saddle:
        return 'saddle'
    elif x > 0 and x < 0.2287:
        return 'maximum'
    else:
        return 'minimum'


def get_hist2d(data, edges_x, edges_y):
    """Bin data in two-dimensional histogram

    Args:
        data (np.ndarray): 2-dimensional array containing data in rows.
        edges_x (array like): bin edges in the x direction.
        edges_y (array like): bin edges in the y direction.

    Returns:
        np.ndarray[int]: array of shape (len(edges_x) - 1, len(edged_x) - 1)) 
            with counts as the entries.

    """
    x_bins = np.digitize(data[:, 0], edges_x)  # bin indices for x
    y_bins = np.digitize(data[:, 1], edges_y)  # bin indices for y
    hist2d = np.zeros([len(edges_y) - 1, len(edges_x) - 1])
    for xb, yb in zip(x_bins, y_bins):
        if xb == 0 or yb == 0 or xb == len(edges_x) or yb == len(edges_y):
            pass
        else:
            hist2d[yb-1, xb-1] += 1
    return hist2d
