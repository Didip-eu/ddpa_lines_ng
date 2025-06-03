# Line Segmentation on charters 

<style>
body { min-width: 90%; }
</style>


Segmentation scripts, Kraken-free:

## Train and validate

Training script for Mask-RCNN (with training set automatically built out of the provided image paths):

```bash
PYTHONPATH=. python3 ./bin/ddp_lineseg.py -img_paths dataset/*.jpg -max_epoch 400 -patience 50 -img_size 1024 -backbone resnet101 -batch_size 4
```

Validate (with validation set automatically built out of the provided image paths):

```bash
PYTHONPATH=. python3 ./bin/ddp_lineseg.py -mode validate -img_paths dataset/*.jpg
```


## Detect and visualize


+ `ddp_line_detect.py` - for line detection on entire page, using an existing layout analysis (`*.seals.pred.json`), if it exists:
  

  ```bash
  PYTHONPATH=. ./bin/ddp_line_detect.py -model_path best.mlmodel -mask_classes Wr.OldText -img_paths data/*.jpg -img_paths data/examples/0042453de0344b72519e093c7b20d593.img.jpg -output_format json
  ```

+ `ddp_lineseg_viewer.py` - reading an existing segmentation file:


  ```bash
  PYTHONPATH=. python3 ./bin/ddp_lineseg_viewer.py  -img_paths data/examples/0042453de0344b72519e093c7b20d593.img.jpg -segfile_suffix lines.pred.json
  ```

  ![](data/examples/0042453de0344b72519e093c7b20d593.png){width=50%}

+ `ddp_lineseg_viewer.py` - for debugging or monitoring purpose, on-the-fly prediction and viewing, using the entire image (no layout analysis):

  ```bash
  PYTHONPATH=. ./bin/ddp_lineseg_viewer.py -model_path best.mlmodel -rescale 1 -img_paths data/examples/0042453de0344b72519e093c7b20d593.Wr_OldText.1.img.jpg
  ```


  ![](data/examples/0042453de0344b72519e093c7b20d593.Wr_OldText_1.png){width=50%}

