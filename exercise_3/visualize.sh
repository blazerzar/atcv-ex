#!/bin/bash
(cd toolkit && python visualize_result.py \
    --workspace_path ../workspace \
    --tracker dcf_tracker \
    --show_gt \
    --sequence drunk
)
