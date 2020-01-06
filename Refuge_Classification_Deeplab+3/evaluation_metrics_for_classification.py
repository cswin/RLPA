
import numpy as np

from file_management import read_csv_classification_results, sort_scores_by_filename, get_labels_from_training_data, save_roc_curve, save_csv_classification_performance, read_gt_labels

from sklearn.metrics import roc_auc_score, roc_curve
from os import path, makedirs
from scipy.interpolate import interp1d


def get_roc_curve(predicted_scores, gt_labels):
    '''
    Computes the ROC curve and the area under it (AUC)

    Input:
        predicted_scores: a 1D numpy array with the scores as provided in the CSV file
        gt_labels: a 1D numpy array with the gt labels (0: healthy, 1: glaucomatous)
    Output:
        auc: the area under the ROC curve

    '''

    # compute the ROC curve
    fpr, tpr, _ = roc_curve(gt_labels, predicted_scores)

    # compute the area under the ROC curve
    auc = roc_auc_score(gt_labels, predicted_scores)

    return tpr, fpr, auc



def get_sensitivity_at_given_specificity(sensitivity, specificity, specificity_reference=0.85):
    '''
    Get the sensitivity for a given specificity reference

    Input:
        sensitivity: sensitivity values
        specificity: specificity values
        [specificity_reference]: reference value for evaluation
    Output:
        sensitivity_value: sensitivity value at the specificity reference
    '''

    # interpolate a continue curve based on sensitivity and specificity
    sensitivity_interp = interp1d(specificity, sensitivity)
    # get the sensitivity value for the specificity reference
    sensitivity_value = sensitivity_interp(specificity_reference)

    return sensitivity_value



def evaluate_classification_results(prediction_filename, gt_folder, output_path=None, is_training=False):
    '''
    Evaluate the results of a classification algorithm

    Input:
        prediction_filename: full path with file name to a .csv file with the classification results
        gt_folder: folder where the ground truth labels are given. If is_training, it should be the path to the Glaucoma / Non-Glaucoma labels
        [output_path]: a folder where the results will be saved. If not provided, the results are not saved
        [is_training]: a boolean value indicating if the evaluation is performed on training data or not
    '''

    # read the prediction filename
    image_filenames, predicted_scores = read_csv_classification_results(prediction_filename)


    # we will treat differently the labels from the training set
    if is_training:
        # we will use the gt folder to retrieve the glaucomatous and no-glaucomatous cases
        gt_filenames, gt_labels = get_labels_from_training_data(gt_folder)
    else:
        # get the filenames and the labels
        gt_filenames, gt_labels = read_gt_labels(path.join(gt_folder, 'GT.xlsx'))
    
    # sort the gt filenames using the same order as predicted
    gt_labels = sort_scores_by_filename(image_filenames, gt_filenames, gt_labels)
    
    # compute the ROC curve
    sensitivity, fpr, auc = get_roc_curve(predicted_scores, gt_labels)
    # compute specificity
    specificity = 1 - fpr
    # print the auc
    print('AUC = {}'.format(str(auc)))

    # get sensitivity at reference value
    sensitivity_at_reference_value = get_sensitivity_at_given_specificity(sensitivity, specificity)
    # print the value
    print('Reference Sensitivity = {}'.format(str(sensitivity_at_reference_value)))

    # if the output path is given, export the results
    if not (output_path is None):

        # create the folder if necessary
        if not path.exists(output_path):
            makedirs(output_path)

        # save the ROC curve
        save_roc_curve(path.join(output_path, 'roc_curve.mat'), sensitivity, fpr, auc)

        # save a CSV file with the reference metrics
        save_csv_classification_performance(path.join(output_path, 'evaluation_classification.csv'), auc, sensitivity_at_reference_value)

    return auc, sensitivity_at_reference_value


if __name__ == '__main__':


    classification_filename = 'test_val_23_5fold_800.csv'
    gt_folder='../data/Annotation-Training400/Disc_Cup_Masks/'
    
    # call the "main" function
    evaluate_classification_results(classification_filename, gt_folder,is_training=True)
    
    