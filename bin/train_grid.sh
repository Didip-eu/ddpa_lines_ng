#!/bin/bash
#
# Train on a grid of hyper-parameters
#
scheduler_factors=".9 .8 .7"
scheduler_patiences="6 8 4"
learning_rates=".0001 .0002 .0005 .00075"
backbones="resnet50 resnet101"

export PYTHONPATH=.
for sf in $scheduler_factors; do
	for sp in $scheduler_patiences; do
		for lr in $learning_rates; do
			for bb in $backbones; do
				saved_model=models/best.$(date +"%m%d%Y").lr${lr:1}.bb${bb}.sf${sf:1}.sp${sp}.mlmodel
				test -f "${saved_model}" && continue
				./bin/ddp_lineseg_train.py -lr ${lr} -backbone ${bb} -scheduler_factor ${sf} -cache_dir dataset/cached -max_epoch 60 -scheduler 1 -scheduler_patience ${sp} -scheduler_cooldown 2 -patience 20 -max_epoch 100 -device cuda -batch_size 6
				test -f best.mlmodel && mv best.mlmodel models/best.$(date +"%m%d%Y").lr${lr:1}.bb${bb}.sf${sf:1}.sp${sp}.mlmodel
				rm -f last.*
			done
		done
	done
done
