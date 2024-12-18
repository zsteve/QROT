#!/bin/bash

for c in $(seq 0 0.1 1); do 
    # python CLEBoolODE_HSC.py $c mult;
    python CLEBoolODE_HSC.py $c none;
done
