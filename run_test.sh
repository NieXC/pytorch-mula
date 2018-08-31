CUDA_VISIBLE_DEVICES=0 python main.py \
                       --evaluate True \
                       --calc-pck True \
                       --calc-miou True \
                       --resume exps/snapshots/mula_lip.pth.tar \
                       --visualization True \
                       #--eval-data dataset/lip/testing_images \
                       #--eval-pose-anno dataset/lip/jsons/LIP_SP_TEST_annotations.json \
                       #--visualization True \
                       #--vis-dir exps/preds/vis_results \
