import argparse
import os

import cv2

from siamfc import TrackerSiamFC
from tools.sequence_utils import VOTSequence, save_results


def evaluate_tracker(
    dataset_path, network_path, results_dir, visualize, redetections=False
):

    sequences = []
    redetection_lengths = []
    with open(os.path.join(dataset_path, 'list.txt'), 'r') as f:
        for line in f.readlines():
            sequences.append(line.strip())

    for sequence_name in sequences:
        tracker = TrackerSiamFC(net_path=network_path)

        print('Processing sequence:', sequence_name)

        bboxes_path = os.path.join(results_dir, '%s_bboxes.txt' % sequence_name)
        scores_path = os.path.join(results_dir, '%s_scores.txt' % sequence_name)

        if os.path.exists(bboxes_path) and os.path.exists(scores_path):
            print('Results on this sequence already exists. Skipping.')
            continue

        sequence = VOTSequence(dataset_path, sequence_name)

        img = cv2.imread(sequence.frame(0))
        gt_rect = sequence.get_annotation(0)
        tracker.init(img, gt_rect)
        results = [gt_rect]
        scores = [[10000]]  # a very large number - very confident at initialization

        if visualize:
            cv2.namedWindow('win', cv2.WINDOW_AUTOSIZE)
        for i in range(1, sequence.length()):
            img = cv2.imread(sequence.frame(i))
            prediction, score = tracker.update(img)
            results.append(prediction)
            scores.append([score])

            if visualize:
                gt_ = sequence.get_annotation(i)
                tl_ = int(round(gt_[0])), int(round(gt_[1]))
                br_ = (
                    int(round(gt_[0] + gt_[2])),
                    int(round(gt_[1] + gt_[3])),
                )
                cv2.rectangle(img, tl_, br_, (0, 255, 0), 2)

                tl_ = (int(round(prediction[0])), int(round(prediction[1])))
                br_ = (
                    int(round(prediction[0] + prediction[2])),
                    int(round(prediction[1] + prediction[3])),
                )
                if score > 0:
                    cv2.rectangle(img, tl_, br_, (0, 0, 255), 2)

                # Draw redetection samples
                if tracker.cfg.long_term:
                    s = tracker.last_size
                    for x, y in tracker.last_samples:
                        tl_ = int(round(y - s / 2)), int(round(x - s / 2))
                        br_ = int(round(y + s / 2)), int(round(x + s / 2))
                        cv2.rectangle(img, tl_, br_, (200, 255, 88))

                cv2.imshow('win', img)
                key_ = cv2.waitKey(10)
                if key_ == 27:
                    exit(0)

        save_results(results, bboxes_path)
        save_results(scores, scores_path)

        redetection_lengths.append(tracker.redetection_lengths)

    if redetections:
        return redetection_lengths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SiamFC Runner Script')

    parser.add_argument(
        "--dataset", help="Path to the dataset", required=True, action='store'
    )
    parser.add_argument(
        "--net", help="Path to the pre-trained network", required=True, action='store'
    )
    parser.add_argument(
        "--results_dir",
        help="Path to the directory to store the results",
        required=True,
        action='store',
    )
    parser.add_argument(
        "--visualize",
        help="Show ground-truth annotations",
        required=False,
        action='store_true',
    )

    args = parser.parse_args()

    evaluate_tracker(args.dataset, args.net, args.results_dir, args.visualize)
