import time
from dataclasses import dataclass
from os import listdir, path

import cv2
import numpy as np
from numpy.typing import NDArray

from ms_tracker import MeanShiftTracker, MSParams
from sequence_utils import VOTSequence


@dataclass
class Failure:
    image: NDArray
    truth: tuple[tuple[int, int], tuple[int, int]]
    detect: tuple[tuple[int, int], tuple[int, int]]


def run_tracker(
    sequence_title: str, parameters: MSParams
) -> tuple[float, list[Failure]]:
    sequence = VOTSequence('vot2014', sequence_title)
    init_frame = 0
    failures = []
    tracker = MeanShiftTracker(parameters)

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

        if o > 0:
            frame_idx += 1
        else:
            # Increase frame counter by 5 and set re-initialization
            frame_idx += 5
            init_frame = frame_idx
            failures.append(
                Failure(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                    bb_to_rect(gt_bb),
                    bb_to_rect(predicted_bbox),
                )
            )

    fps = sequence.length() / time_all
    return fps, failures


def bb_to_rect(
    bb: tuple[int, int, int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Convert bounding box to cv2 rectangle."""
    tl = np.round(bb[:2]).astype(int)
    br = np.round([bb[0] + bb[2] - 1, bb[1] + bb[3]]).astype(int)
    return (tl[0], tl[1]), (br[0], br[1])


def main() -> None:
    all_sequences = sorted(
        [name for name in listdir('vot2014') if path.isdir(path.join('vot2014', name))]
    )

    all_failures = 0
    for sequence in all_sequences:
        parameters = MSParams()
        fps, failures = run_tracker(sequence, parameters)
        all_failures += len(failures)
        print(f'{sequence}: {fps:.2f} fps, {len(failures)} failures')
    print('Total failures:', all_failures)


if __name__ == '__main__':
    main()
