"""Simple transformation script

"""

import os, sys
import argparse
import numpy as np


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infpath", type=str, required=True)
    parser.add_argument("-o", "--outfpath", type=str, required=True)
    parser.add_argument("-t", "--transform", type=str, required=True)
    return parser.parse_args(args)


def transform1(x):
    assert x.ndim == 3, f"Expected 3-dimensional input array. Got shape {x.shape}"
    assert x.shape[-1] == 2, f"Expected data to consist of 2-dimensional points."
    x1 = x[...,0]
    x2 = x[...,1]
    y1 = x1
    y2 = x2
    y3 = x1**2 + x2**2
    y = np.stack([y1, y2, y3], axis=-1)
    return y


def transform2(x):
    assert x.dim == 3, f"Expected 3-dimensional input array. Got shape {x.shape}"
    assert x.shape[-1] == 2, f"Expected data to consist of 2-dimensional points."
    x1 = x[...,0]
    x2 = x[...,1]
    y1 = x1
    y2 = x2
    y3 = x1**2 + x2**2
    y = np.stack([y1, y2, y3], axis=-1)
    return y


def main(args):
    infpath = args.infpath
    outfpath = args.outfpath
    transform = args.transform
    
    transform_func = {
        "transform1": transform1,
        # "transform2": transform2,
    }[transform]

    x_in = np.load(infpath, allow_pickle=True)
    x_out = transform_func(x_in)
    np.save(outfpath, x_out)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)

