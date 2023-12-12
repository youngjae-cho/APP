

![image](https://github.com/youngjae-cho/APP/assets/58834857/51e1807d-6dcb-433d-b7ed-acc3a6628ac9)

# APP
This folder contains the implementation of the APP method on prompt learning.

This code is built on top of [CoOp](https://github.com/KaiyangZhou/CoOp), [PLOT](https://github.com/CHENGY12/PLOT), and [dassl](https://github.com/KaiyangZhou/Dassl.pytorch#installation) .


## Install Dataset
Please follow the instructions [DATASETS.md](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to construct the datasets.


## Run Scripts


The running scripts are in `scripts/`. `cd ./scripts` and change the `your_data_path` and `your_work_path` in `scripts/main.sh`
Then, you can run the commands `bash main.sh DATASET N` under `scripts/`.

`DATASET` takes as input a dataset name, like `caltech101`. 

`N` is the number of prompts, such as `4`.

## Performance

![image](https://github.com/youngjae-cho/APP/assets/58834857/1ba92412-219f-4e08-9471-ec7e15f42bb8)
![image](https://github.com/youngjae-cho/APP/assets/58834857/2c0dd9ba-f234-4dfe-8b05-f23998381663)
