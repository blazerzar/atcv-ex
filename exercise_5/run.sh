#!/bin/bash
rm -f results/* && DCF_LONG_TERM=1 python run_tracker.py --dataset dataset --net siamfc_net.pth \
    --results_dir results --visualize
