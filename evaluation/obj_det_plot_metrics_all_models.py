import sys, os
sys.path.append(os.getcwd())
from alficore.evaluation.sdc_plots.obj_det_plot_metrics_all_models import obj_det_plot_metrics

def obj_det_analysis(argv):

    # Pay attention not to add an unnecessary / at the end
    exp_folder_paths = {#"frcnn+CoCo":{
                        #"neurons":{"path" :"result_files/frcnn_torchvision_1_trials/neurons_injs/per_batch/objDet_20250205-152419_1_faults_[0,7]_bits/coco", "typ":"no_resil"},
                        #"weights":{"path" :"result_files/result_files_paper/frcnn_torchvision_3_trials/weights_injs/per_batch/objDet_20250204-164950_1_faults_[0,9]_bits/coco", "typ":"no_resil"}
                        #},
            #             },
              "Yolo+Coco":{
                         "neurons":{"path": "result_files/yolo_torchvision_1_trials/neurons_injs/per_batch/objDet_20250205-154712_1_faults_[0,7]_bits/coco", "typ":"ranger"}, 
                         #"weights":{"path" :"path", "typ":"no_resil"}
                         }
            }
    obj_det_plot_metrics(exp_folder_paths=exp_folder_paths)

if __name__ == "__main__":
    obj_det_analysis(sys.argv)