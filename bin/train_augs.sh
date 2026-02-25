#!/bin/bash
#
# Train on a grid of hyper-parameters
#
bb="resnet101"
sf=.9
sp=6
lr=0.00075
datasets="be2ddcea38b216dc9daf42d1144137ba.0 be2ddcea38b216dc9daf42d1144137ba.2 c964785fee8175ddf707223567464d2e.3 c964785fee8175ddf707223567464d2e.5"
export PYTHONPATH=.
for ds in $datasets; do
	saved_model=models/best.lr${lr:1}.bb${bb}.sf${sf:1}.sp${sp}.ds${ds}.mlmodel
	test -f "${saved_model}" && continue
	echo "bb=${bb} sf=${bb} sp=${sp} lr=${lr} ds=${ds}"
	./bin/ddp_lineseg_train.py -lr ${lr} -backbone ${bb} -scheduler_factor ${sf} -cache_dir ../data/cached.$ds -scheduler 1 -scheduler_patience ${sp} -scheduler_cooldown 2 -patience 20 -max_epoch 100 -device cuda -batch_size 12
	test -f best.mlmodel && mv best.mlmodel $saved_model
	rm -f last.*
done
