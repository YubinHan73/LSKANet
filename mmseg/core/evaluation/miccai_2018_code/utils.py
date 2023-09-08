import os
import numpy as np
import matplotlib.pyplot as plt
import re


plt.style.use('ggplot')

"""

For instrument segmentation.
We need 3 types of 'results'
Mean IOU foreground/background
Mean IOU shaft/wrist/clasper/background
Mean IOU type (cannot be broken up per dataset)

Each 'Submission' is associated with a user and contains a listf of IOUs or each possible class for each dataset



"""


class Label:

    def __init__(self, json_data):

        self.name = json_data['name']
        self.color = json_data['color']
        self.classid = json_data['classid']


class Submission:

    def __init__(self, name):

        self.dataset_results = []
        self.name = name
    
    def get_number_of_datasets(self):
        return len(self.dataset_results)

    def get_number_of_images_for_dataset(self, dataset_idx):
        return self.dataset_results[dataset_idx].size

    def get_short_name(self):
        return self.name.split(" ")[0].lower()[0] + self.name.split(" ")[1].lower()[0:4]

    def get_label_name(self):
        return self.name.split(" ")[0][0] + " " + self.name.split(" ")[1]

    def compute_scores_across_datasets(self):
        
        self.mean_iou_per_dataset = []
        for dset in self.dataset_results:
            results = [fname_im[1] for fname_im in dset.results if fname_im[1] is not None ]
            if len(results) > 0:
                self.mean_iou_per_dataset.append(np.mean(results))
            else:
                self.mean_iou_per_dataset.append(None)
            

        total_frames = sum([len(f.results) for f in self.dataset_results])
        if total_frames == 0:
            self.mean_iou_all_datasets = None
            return False

        dset_lens = [float(len(f.results))/total_frames for f in self.dataset_results]

        # if they have any nans in the mean score, they don't get a score for this problem
        if np.inf in self.mean_iou_per_dataset or np.nan in self.mean_iou_per_dataset:
            self.mean_iou_all_datasets = None
        else:
            w = [f for f in dset_lens if f > 0]
            self.mean_iou_all_datasets = np.average([f for f in self.mean_iou_per_dataset if f is not None], weights=w)#np.mean(self.mean_iou_per_dataset)
        return self.mean_iou_all_datasets is not None
    
    def get_dataset_results(self, dataset_idx):
        return self.dataset_results[dataset_idx]


class ImageResults:

    def __init__(self, ground_truth_image, submission_image, error_function):

        self.error = error_function(submission_image, ground_truth_image)
        self.num_pixels = ground_truth_image.shape[0] * ground_truth_image.shape[1]


class DatasetResults:

    def __init__(self, dataset_name, error_function):

        self.name = dataset_name
        self.results = []
        self.error_function = error_function

    def is_multiclass_results(self):
        return isinstance(self.error_function, MulticlassIntersectionOverUnionError)

    def add_results_for_frame(self, file_name, ground_truth_image, submission_image):

        if ground_truth_image is None:
            raise Exception("Error, ground truth image should never be null!")
        else:    
            error = self.error_function.get_error(ground_truth_image, submission_image)
            if error is not None:
                # we return None is there are no pixels for the label in the image (ground truth)
                self.results.append((file_name, error))
                return True
            return False

    def get_per_frame_iou(self):
        return [fname_im[1] for fname_im in self.results]

    def get_frame_numbers(self):
        return [int(re.findall(r'\d+', x[0])[0]) for x in self.results ]

    def is_good(self):
        return np.sum([ f[1] for f in self.results]) > 0


def get_intersection_over_union(overlap, prediction_not_overlap, ground_truth_not_overlap):

    if (ground_truth_not_overlap + overlap) == 0:
        # this occurs when there is no pixels for ground truth class in image - no score in this image
        return None

    return float(overlap) / (prediction_not_overlap + ground_truth_not_overlap + overlap)


class MulticlassIntersectionOverUnionError:

    vals = []
    orig_vals = []

    def __init__(self, vals):
        MulticlassIntersectionOverUnionError.vals = vals
        MulticlassIntersectionOverUnionError.orig_vals = vals
        
    @staticmethod
    def reset_orig_vals():
        MulticlassIntersectionOverUnionError.vals = MulticlassIntersectionOverUnionError.orig_vals
    
    @staticmethod
    def switch_on_val(val):
        MulticlassIntersectionOverUnionError.vals = [val]

    def get_error(self, ground_truth_frame, entry_frame):

        ious = []
        
        for val in MulticlassIntersectionOverUnionError.vals:
            
            if val.classid == 0:
                continue

            true_positive_count = np.sum ( (ground_truth_frame == val.classid) & (entry_frame == val.classid) )
            true_negative_count = np.sum ( (ground_truth_frame != val.classid) & (entry_frame != val.classid) )
            false_positive_count = np.sum ( (ground_truth_frame != val.classid) & (entry_frame == val.classid) )
            false_negative_count = np.sum ( (ground_truth_frame == val.classid) & (entry_frame != val.classid) )

            iou = get_intersection_over_union( true_positive_count, false_positive_count, false_negative_count)
            if iou is None:
                # only count IoU for mean for classes actually in this frame
                continue
            else:
                ious.append( iou )

        if len(ious) == 0:
            return None
        return np.mean(ious)

