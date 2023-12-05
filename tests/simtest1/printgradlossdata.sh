#!/bin/bash

dt=$1

d1_wls_arglist=("0 0 0 1 1 0 1 1 0 1")
d2_wls_arglist=("1 1 1 0 0 1 0 0 0 1")
d3_wls_arglist=("2 0 2 1 3 0 3 1 1 0")
d4_wls_arglist=("3 0 3 1 2 0 2 1 1 0")

echo "d1:"
for args in "${d1_wls_arglist[@]}"; do
    wolframscript tests/simtest1/rungradloss.wls $args $dt |  sed 's/{/[/g; s/}/]/g'
done;

echo "d2:"
for args in "${d2_wls_arglist[@]}"; do
    wolframscript tests/simtest1/rungradloss.wls $args $dt |  sed 's/{/[/g; s/}/]/g'
done;

echo "d3:"
for args in "${d3_wls_arglist[@]}"; do
    wolframscript tests/simtest1/rungradloss.wls $args $dt |  sed 's/{/[/g; s/}/]/g'
done;

echo "d4:"
for args in "${d4_wls_arglist[@]}"; do
    wolframscript tests/simtest1/rungradloss.wls $args $dt |  sed 's/{/[/g; s/}/]/g'
done;
