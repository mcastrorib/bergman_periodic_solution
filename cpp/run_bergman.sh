#!/bin/bash
# This scrip runs the BERGMAN_SOLUTION program

# run cpp program (results are saved at ~/db/temp)
./bergman

# run python script to visualize results
python3 .src/datavis/bergman_plot.py




