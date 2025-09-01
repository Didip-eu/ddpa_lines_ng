
# Evaluation scripts and outputs

## Line-based scores for given IoU (Icdar 2017=.75) over box and mask thresholds

File `eval_out.tsv` a file of the form:

```
IoU	B-Thr	M-Thr	TP	FP	FN	Precision		Recall			Jaccard			F1
0.75	0.85	0.25	3917.0	730.0	141.0	0.8429094039165053	0.9652538196155742	0.8180868838763575	0.899942561746123
0.75	0.85	0.3	4100.0	552.0	150.0	0.88134135855546	0.9647058823529412	0.85381091211995	0.9211413165580769
...
```
Generated with:

```
PYTHONPATH=.
# all pixels
PYTHONPATH=.; for bt in .7 .75 .8 0.85 0.9 0.95 0.97 0.98 ; do for mt in 0.25 0.3 0.35 .40 .45 .5 .55 .6 .65; do ./bin/ddp_lineseg_eval.py -img_paths dataset/val/*.jpg  -patch_size 1024 -mask_threshold $mt -box_threshold $bt -output_file_name '>>eval_out.tsv' -cache_predictions 1 -output_root_dir evaluation/2.0 -save_file_scores 1; done; done
```

With the options used above, the evalutation script generates two additional sets of files:

+ The prediction files (options '-cache_predictions'), i.e. that store the segmentation results as tensors (`*.pt`): they do not depend on any box or mask threshold and can be reused for each of subsequent iteration of the loop, with different post-processing parameters.
+ The individual scores for each image ('-save_file_scores' option), for a given combination of thresholds (`file_scores_*.tsv`).

Or, with foreground pixels only:

```
PYTHONPATH=.; for bt in .7 .75 .8 0.85 0.9 0.95 0.97 0.98 ; do for mt in 0.25 0.3 0.35 .40 .45 .5 .55 .6 .65; do ./bin/ddp_lineseg_eval.py -img_paths dataset/val/*.jpg -foreground_only 1  -patch_size 1024 -mask_threshold $mt -box_threshold $bt -output_file_name '>>eval_out_foreground.tsv' -cache_predictions 1 -output_root_dir evaluation/2.0 -save_file_scores 0; done; done
```
Each set of output files is stored in a different subdirectory, whose name is the MD5 of the model file used for the predictions (accordingly, even if it is not loaded in memory for the job, this model file dictates which subdirectory to check for the cached predictions). For instance, the following, compound command generates both kinds of evaluation files (with and without polygon binarization) for two different models:

```
PYTHONPATH=.; for mp in models/{best_cached_patches,best_no_scheduler_62_iterations}.mlmodel ; do for bt in .7 .75 .8 0.85 0.9 0.95 0.97 0.98 ; do for mt in  0.25 0.3 0.35 .40 .45 .5 .55 .6 .65; do echo "$(md5sum $mp) $bt $mt"; ./bin/ddp_lineseg_eval.py -img_paths dataset/val/*.jpg  -patch_size 1024 -mask_threshold $mt -box_threshold $bt -output_file_name '>>eval_out.tsv' -model_path $mp -cache_predictions 1 -output_root_dir evaluation/2.0 -save_file_scores 1; done; done ; done ;   for mp in models/{best_cached_patches,best_no_scheduler_62_iterations}.mlmodel ; do for bt in .7 .75 .8 0.85 0.9 0.95 0.97 0.98 ; do for mt in  0.25 0.3 0.35 .40 .45 .5 .55 .6 .65; do echo "$(md5sum $mp) $bt $mt"; ./bin/ddp_lineseg_eval.py -img_paths dataset/val/*.jpg -foreground_only 1  -patch_size 1024 -mask_threshold $mt -box_threshold $bt -output_file_name '>>eval_out_foreground.tsv' -model_path $mp -cache_predictions 1 -output_root_dir evaluation/2.0 -save_file_scores 0; done; done ; done
```

Headers are handled internally by the Python script (if a file is written more than once, the header is not duplicated).

## Recall-precision (mAP) over a range of IoU for given box and mask thresholds:


Eg. the file `recall_precision_test_0.5-0.95.tsv`:

```
IoU	B-Thr	M-Thr	TP	FP	FN	Precision	Recall	Jaccard	F1
0.5	0.95	0.45	5782.0	66.0	192.0	0.9887140902872777	0.9678607298292601	0.9572847682119205	0.9781762815090509
0.55	0.95	0.45	5738.0	72.0	230.0	0.9876075731497418	0.9614611260053619	0.95	0.9743589743589743
...
0.95	0.95	0.45	256.0	4901.0	2324.0	0.049641264300950165	0.09922480620155039	0.0342200240609
```

has been generated with:

```
PYTHONPATH=.; b=0.95; mt=.5; iou=0.5; while [[ $(echo "$iou < 1"|bc) -eq 1 ]] ; do echo "IoU=$iou" ; ./bin/ddp_lineseg_eval.py -img_paths dataset/test/*.jpg  -patch_size 1024 -mask_threshold $mt -box_threshold $bt -icdar_threshold $iou -cache_predictions 1 -output_root_dir evaluation/2.0 -output_file_name '>>recall_precision_test_0.5-0.95.tsv' ; iou=$( echo "$iou+.5"|bc ) ; done;
```

