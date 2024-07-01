#!/bin/bash
rm -rf workspace/results/dcf/*
(cd toolkit && python evaluate_tracker.py \
    --workspace_path ../workspace \
    --tracker dcf_tracker
)
