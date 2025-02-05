# Copyright 2022 Intel Corporation.
# SPDX-License-Identifier: MIT

import sys, os
sys.path.append(os.getcwd())
from alficore.evaluation.sdc_plots.obj_det_analysis import obj_det_analysis as objdet_analysis

def obj_det_analysis(argv):

    # Pay attention not to add an unnecessary / at the end
    exp_folder_paths = [
                    "result_files/frcnn_torchvision_1_trials/neurons_injs/per_batch/objDet_20250205-152419_1_faults_[0,7]_bits/coco"
                    ]
    resil_methods = ["no_resil"]*len(exp_folder_paths)
    objdet_analysis(exp_folder_paths=exp_folder_paths, resil_methods=resil_methods, num_threads=1)

if __name__ == "__main__":
    obj_det_analysis(sys.argv)