#!/bin/bash
(cd toolkit && python calculate_measures.py \
    --workspace_path ../workspace \
    --tracker dcf_tracker
)
