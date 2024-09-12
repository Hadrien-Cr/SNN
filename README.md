prepare dataset:
-  go to the [download page](https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794) of DVS Gesture and download all 4 files
- place the file in your working directory following the arborescence
```
working_dir
    .
    |-- DVS128Gesture
        .
        |--  download
            .
            |-- DvsGesture.tar.gz
            |-- LICENSE.txt
            |-- README.txt
            `-- gesture_mapping.csv
```


run tutorial (dvs gesture no delays) from https://github.com/fangwei123456/spikingjelly/blob/0.0.0.0.14/spikingjelly/activation_based/examples/classify_dvsg.py: 
```
python3 main.py -no-delays
```

run dvs gesture with learnable delays (https://github.com/Thvnvtos/SNN-delays/blob/master/snn_delays.py adapted to DVS Gesture): (change config.py if needed for parameters and hyperparameters)
```
python3 main.py
```