#!/bin/bash --login
eval "$(conda shell.bash hook)"

ch="abc" # ['abc', 'harmony']
if [ "$ch" = "abc" ]; then
    # Phase 1: ABC
    conda activate abc
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
    cd ./Phase_1-ABC
    python main.py --output_dir saved_data
    cp saved_data/corrected.h5ad ../dataset/
    conda deactivate
elif [ "$ch" = "harmony" ]; then
    # Phase 1: Harmony
    conda activate usb
    cd ./Phase_1-Harmony
    python main.py --output_dir saved_data
    cp saved_data/corrected.csv ../dataset/
    conda deactivate
else
    echo "Please enter valid pre-processing option."
fi

# Phase 2: Commingle
cd ../Phase_2-Commingle
conda activate usb
if [ "$ch" = "abc" ]; then
    python main.py --first_run --dataset_path ../dataset/corrected.h5ad
    python clam_eval.py --checkpoint s_4_best_checkpoint.pt
else
    python main.py --first_run --dataset_path ../dataset/corrected.csv --harmony
    python clam_eval.py --checkpoint s_4_best_checkpoint.pt --harmony
fi
cp saved_data/weights.csv ../dataset/
cp saved_data/corrected.csv ../dataset/

# Phase 3: Postprocessing
cd ../Phase_3-Downstream
python main.py --attention --base_path ../dataset/ --weight_path weights.csv --dataset_name Micro_PVM
conda deactivate 

