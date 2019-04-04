# prohib-detection

## Quickstart

You will need to install the Tensorflow Object Detection API (https://github.com/tensorflow/models/tree/master/research/object_detection)

To do that git clone this repository : https://github.com/tensorflow/models.git

Git clone this repository (https://github.com/paulesta55/prohib-detection.git) at the root of Tensorflow Models repository (under 'models')

Then you will need to follow the instructions here : https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

Download this archive (http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz) and untar its content under '/prohib-detection/pre-trained-model/faster-rcnn'.

Download this archive (http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz) and untar its content under '/prohib-detection/pre-trained-model/ssd'.

#### dataset
https://drive.google.com/open?id=1ajmN_w9rcqmWojDl_6j2L-GtwUXTLKnR

unzip its content under '/prohib-detection/images/'

#### Generate tf records 

from  '/prohib-detection/'

```bash
$ python generate_tfrecord.py --img_path=images/train --label='red-circle' --csv_input=annotations/train_ul.csv --output_path=annotations/train_ul.record
$ python generate_tfrecord.py --img_path=images/test --label='red-circle' --csv_input=annotations/test_ul.csv --output_path=annotations/test_ul.record
```

#### Train the networks

##### To train the ssd_inceptionv2 network:

from  '/prohib-detection/'

```bash
$ python train.py --logtostderr --train_dir=training --pipeline_config_path=training/ssd_inception_v2_coco.config
```

##### To train the faster-rcnn_resnet50 network :

from  '/prohib-detection/'

```bash
$ python train.py --logtostderr --train_dir=training2 --pipeline_config_path=training2/faster_rcnn_resnet50_coco.config
```

You can monitor the training with Tensorboard using 'training' and 'training2' as 'logdir'

#### Evaluate 

##### For ssd_inceptionv2 network

```bash
$ python eval.py --logtostderr --checkpoint_dir=training --eval_dir=eval --pipeline_config_path=training/ssd_inception_v2_coco.config
```

##### For faster-rcnn_resnet50 network

```bash
$ python eval.py --logtostderr --checkpoint_dir=training2 --eval_dir=eval2 --pipeline_config_path=training2/faster_rcnn_resnet50_coco.config
```
You can monitor the evaluation and see results with Tensorboard using 'eval' and 'eval2' as 'logdir'
