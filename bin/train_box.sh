#!/bin/bash
#
# Train on a grid of hyper-parameters
#
dv=$1
resume_dir=$2
ds=$3
bs=$4
bb="resnet101"
sf=$5
sp=$6
lr=$7
echo $resume_dir $ds $bs $sf $sp $lr
export PYTHONPATH=.
saved_model=${resume_dir}/best.lr${lr:1}.bb${bb}.sf${sf}.sp${sp}.bs${bs}.ds${ds}.mlmodel
test -f "${saved_model}" && continue
echo "bb=${bb} sf=${sf} sp=${sp} lr=${lr} ds=${ds} bs=${bs}"
./bin/ddp_lineseg_train.py -lr ${lr} -backbone ${bb} -scheduler_factor ${sf} -cache_dir ~/extra_data_storage/ddpa_lines_cached_datasets/cached.$ds -scheduler 1 -scheduler_patience ${sp} -scheduler_cooldown 2 -patience 15 -max_epoch 100 -device ${dv} -batch_size ${bs} -resume_dir ${resume_dir}
test -f ${resume_dir}/best.mlmodel && mv ${resume_dir}/best.mlmodel $saved_model
rm -f ${resume_dir}/last.*
