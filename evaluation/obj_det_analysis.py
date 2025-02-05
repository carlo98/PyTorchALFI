# Copyright 2022 Intel Corporation.
# SPDX-License-Identifier: MIT

import sys, os
sys.path.append(os.getcwd())
from alficore.evaluation.sdc_plots.obj_det_analysis import obj_det_analysis as objdet_analysis

def obj_det_analysis(argv):

    exp_folder_paths = [
                    "result_files/result_files_paper/frcnn_torchvision_3_trials/weights_injs/per_batch/objDet_20250204-164950_1_faults_[0,9]_bits/coco"
                    ]
    resil_methods = ["no_resil"]*len(exp_folder_paths)
    objdet_analysis(exp_folder_paths=exp_folder_paths, resil_methods=resil_methods, num_threads=1)

if __name__ == "__main__":
    obj_det_analysis(sys.argv)