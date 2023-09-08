import cv2
import os
from .utils import Submission, Label, MulticlassIntersectionOverUnionError, DatasetResults
import json
import matplotlib.pyplot as plt
import re
import numpy as np

base_dir = os.path.dirname(__file__)
NUM_TEST_DIRS = 4
NUM_CLASSES = 12


def get_error_and_labels():

    with open(os.path.join(base_dir, 'labels.json')) as data_file:
        data = json.load(data_file)
        labels = [Label(d) for d in data['classes']]
        error_function = MulticlassIntersectionOverUnionError(labels)

    return (error_function, labels)


def process(error_function, results, gt_seg_maps):

    submissions = [Submission('submission0')]
    dataset = 'dataset0'
    if not isinstance(gt_seg_maps, list):
        gt_seg_maps = list(gt_seg_maps)

    for entry in submissions:
        entry.dataset_results.append(DatasetResults(dataset, error_function))

    # max_image = 3
    # image_now = 0
    for frame in range(len(results)):
        # image_now += 1
        ground_truth_image = gt_seg_maps[frame]

        for entry in submissions:
            submission_image = results[frame]
            entry.dataset_results[-1].add_results_for_frame(str(frame), ground_truth_image, submission_image)
        # if image_now == max_image:
        #     break

    return submissions


def get_submission_miou(submissions):

    # work out which entrants have submissions
    submission = submissions[0]
    submissions_with_entries = []
    if submission.compute_scores_across_datasets():
        submissions_with_entries.append(submission)

    if len(submissions_with_entries) == 0:
        return

    return submission.mean_iou_all_datasets


def get_miou(results, gt_seg_maps):
    # miou over all classes (first) and images (second)
    error_and_labels = get_error_and_labels()
    submissions = process(error_and_labels[0], results, gt_seg_maps)
    miou = get_submission_miou(submissions)
    return miou
