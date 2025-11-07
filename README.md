# Line segmentation (Mask-RCNN)



Line segmentation scripts.


## 1. Detect


 `ddp_line_detect.py` - for line detection on entire page, using an existing [layout analysis](https://github.com/anguelos/ddpa_layout.git) (`*.layout.pred.json`), if it exists. The following command runs a model trained on 1024x1024 patches, using the CPU as the default computing device:


```sh
export FSDB_ROOT=/tmp/data/fsdb_subset; export PYTHONPATH=.; echo $FSDB_ROOT/*/*/*/*.img.*p*g | xargs python3 ./bin/ddp_line_detect.py -model_path ./best.mlmodel -line_height_factor 1.3 -output_format json -img_paths
```

Remarks:

+ The `-model_path` parameter should point at the weight file.
+ The command above yields a JSON file; the default output is a JSON dictionary on the _standard output_ (or, explicitly: `-output_format stdout`).
+ The `-line_height_factor` parameter determines how much of the line should be serialized, with respect to the detected 'x-height' for the line component. Eg. With the factor above (1.3), if the detected core line is 10-pixel high, the resulting polygon is a strip that is 13-pixel thick. Default: 1.0 ($\approx$ core line only).
+ By default, the JSON serialization includes non-standard line attributes, such as the line x-height (in pixels) and the centerline points: they are not part of the PageXML specs and are omitted from the corresponding output format (see `-output_format xml` format option).
+ To run on the GPU, pass the `-device gpu` option.

## 2. Visualize

+ `ddp_lineseg_viewer.py` - reading an existing segmentation file:


  ```sh
  PYTHONPATH=. python3 ./bin/ddp_lineseg_viewer.py -img_paths data/examples/0042453de0344b72519e093c7b20d593.img.jpg -segfile_suffix .lines.pred.json
  ```

  ![](data/examples/0042453de0344b72519e093c7b20d593.png)

+ `ddp_lineseg_viewer.py` - for debugging or monitoring purpose, on-the-fly prediction and viewing, using the entire image (no layout analysis):

  ```sh
  PYTHONPATH=. ./bin/ddp_lineseg_viewer.py -model_path best.mlmodel -rescale 1 -img_paths data/examples/0042453de0344b72519e093c7b20d593.Wr_OldText.1.img.jpg
  ```

  ![](data/examples/0042453de0344b72519e093c7b20d593.Wr_OldText_1.png)


## 3. Train and validate

Training script for Mask-RCNN, page-wide (with training set automatically built out of the provided image paths):

```sh
PYTHONPATH=. python3 ./bin/ddp_lineseg.py -img_paths dataset/*.jpg -max_epoch 400 -patience 50 -img_size 1024 -backbone resnet101 -batch_size 4
```

Validate (with validation set automatically built out of the provided image paths):

```sh
PYTHONPATH=. python3 ./bin/ddp_lineseg.py -mode validate -img_paths dataset/*.jpg
```

Patch-based training:

```sh
PYTHONPATH=. python3 ./bin/ddp_lineseg.py -img_paths dataset/*.jpg -max_epoch 400 -patience 50 -img_size 1024 -backbone resnet101 -batch_size 4 -train_style patch
```

## More examples

The segmentation pipeline is still in progress: [cases to ponder upon](data/curiosities/README.md).
