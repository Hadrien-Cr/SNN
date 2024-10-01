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



# Results

| **Configuration**                                 | **Max Acc** | **Remarks**                                                        |
|---------------------------------------------------|-------------|--------------------------------------------------------------------|
| **no-delays**                                     | 0.955       | Train accuracy goes up to 0.99                                     |
| **sanity-check**                                  | 0.948       | Same as no-delays                                                  |
| **random-delays-not-learnable**                   | 0.770       |                                                                    |
| **v1-learnable-delays-init-right**                | 0.934       | The repartition of delays does not change over time                |
| **v1-learnable-delays-init-random**               | 0.799       | A clear shift in delay distribution over time                      |
| **gauss-learnable-delays-init-right-maxdelay11**  | 0.798       | No significant difference compared to `maxdelay5`                  |
| **gauss-learnable-delays-init-right-maxdelay5**   | 0.792       | No significant difference compared to `maxdelay11`                 |
| **gauss-learnable-delays-init-random-maxdelay11** | 0.760       | Delays shift, but do not accumulate on bounds                      |
| **gauss-learnable-delays-init-random-maxdelay5**  | 0.764       | Similar to `maxdelay11`, with delay distribution shifting          |