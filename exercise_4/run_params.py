import time
from os import listdir, path

import cv2
import numpy as np

from particle_filter import ParticleFilterTracker, PFParams
from sequence_utils import VOTSequence


def run_tracker(sequence_title: str, parameters: PFParams) -> tuple[int, float, float]:
    sequence = VOTSequence('vot2014', sequence_title)
    init_frame, failures, overlap = 0, 0, []
    tracker = ParticleFilterTracker(parameters)

    time_all = 0.0
    # Tracking loop - goes over all frames in the video sequence
    frame_idx = 0
    while frame_idx < sequence.length() and frame_idx < len(sequence.gt):
        img = cv2.imread(sequence.frame(frame_idx))
        # Initialize or track
        if frame_idx == init_frame:
            # Initialize tracker (beginning of the sequence or after failure)
            t_ = time.time()
            tracker.initialize(
                img, sequence.get_annotation(frame_idx, type='rectangle')
            )
            time_all += time.time() - t_
            predicted_bbox = sequence.get_annotation(frame_idx, type='rectangle')
        else:
            # Track on current frame - predict bounding box
            t_ = time.time()
            predicted_bbox = tracker.track(img)
            time_all += time.time() - t_

        # Calculate overlap (needed to determine failure of a tracker)
        gt_bb = sequence.get_annotation(frame_idx, type='rectangle')
        o = sequence.overlap(predicted_bbox, gt_bb)
        overlap.append(o)

        if o > 0:
            frame_idx += 1
        else:
            # Increase frame counter and set re-initialization
            frame_idx += 1
            init_frame = frame_idx
            failures += 1

    fps = sequence.length() / time_all
    return failures, np.mean(overlap), fps


def eval_vot(params: PFParams, verbose: bool = False) -> tuple[int, float, float]:
    all_sequences = sorted(
        [name for name in listdir('vot2014') if path.isdir(path.join('vot2014', name))]
    )

    all_failures = 0
    overlaps: list[float] = []
    fpss: list[float] = []
    for sequence in all_sequences:
        failures, overlap, fps = run_tracker(sequence, params)

        all_failures += failures
        overlaps.append(overlap)
        fpss.append(fps)
        overlap = np.round(overlap, 2)

        if verbose:
            print(f'{sequence}: {fps:.2f} fps, {failures} failures, {overlap} IoU')

    return all_failures, np.mean(overlaps), np.mean(fpss)  # type: ignore


def main() -> None:
    failures, overlap, fps = eval_vot(PFParams(), verbose=True)
    print('Total failures:', failures)
    print('Average overlap:', np.round(overlap, 2))
    print('Average FPS:', np.round(fps, 2))


if __name__ == '__main__':
    main()
