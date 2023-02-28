# DiffuseExpand (expanding dataset for 2D medical image segmentation using diffusion models)

### Getting Started

#### Build Environment
First, download our repo and then enter our repo
```bash
cd DiffuseExpand
```

For environmental establishment, we include ```.yaml``` files.

If you have an RTX 30XX GPU (or newer), run

```bash
conda env create -f requirements_11_3.yaml
```

If you have an RTX 20XX GPU (or older), run

```bash
conda env create -f requirements_10_2.yaml
```

You can then activate your  conda environment with

```bash
conda activate diffuseexpand
```
---

#### Download pre-training checkpoints

```bash
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt
```

#### Download Datasets
The datasets used in our work are COVID-19 and CGMH Pelvis, please download them from the website and remember the path to the corresponding dataset:

[COVID-19](https://github.com/ieee8023/covid-chestxray-dataset)
[CGMH Pelvis](https://www.kaggle.com/datasets/tommyngx/cgmh-pelvisseg)

---

#### Fine-tune diffusion model (Stage I)
We used 8 Tesla A100 GPU for the experiment, with a batchsize of 2 on each Tesla A100 GPU. For COVID-19, we run
```bash
mkdir ./stage2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python stage1_train.py --dataset COVID19 \
--data_path /path/to/covid-chestxray-dataset/image --csv_path /path/to/covid-chestxray-dataset/metadata.csv \
--save_path ./stage2 --unet_ckpt_path ./256x256_diffusion.pt --cuda_devices 0,1,2,3,4,5,6,7
```
After that, we can get `./stage2/model_stage2_covid_30000.pt`.

For CGMH Pelvis, the first step is to change the name of the checkpoint to `model_stage2_cgmh_{self.resume_step + self.step}.pt` in `utils/train_utils.py:line 219`. Then, we run
```bash
mkdir ./stage2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python stage1_train.py --dataset CGMH \
--data_path /path/to/CGMH_PelvisSegment \
--save_path ./stage2 --unet_ckpt_path ./256x256_diffusion.pt --cuda_devices 0,1,2,3,4,5,6,7
```
After that, we can get `./stage2/model_cgmh_covid_30000.pt`.

---

#### Train Segmenter (Stage II)
We used 2 Tesla A100 GPU for the experiment, with a batchsize of 8 on each Tesla A100 GPU. For COVID-19, we run
```bash
mkdir ./stage3
CUDA_VISIBLE_DEVICES=0,1 python stage2_train.py --dataset COVID19 \
--data_path /path/to/covid-chestxray-dataset/image --csv_path /path/to/covid-chestxray-dataset/metadata.csv \
--save_path ./stage3 --cuda_devices 0,1
```
After that, we can get `./stage3/stage3_covid19_model_10000.pt`.

For CGMH Pelvis, the first step is to change the name of the checkpoint to `stage3_cgmh_model_{step}.pt` in `stage2_train.py:line 355`. Then, we run
```bash
mkdir ./stage3
CUDA_VISIBLE_DEVICES=0,1 python stage2_train.py --dataset CGMH \
--data_path /path/to/CGMH_PelvisSegment \
--save_path ./stage3 --cuda_devices 0,1
```
After that, we can get `./stage3/stage3_cgmh_model_10000.pt`.

---

#### Synthesize Image-Mask pairs (Stage III)

For synthesizing Image-Mask pairs with COVID-19, we run
```bash
mkdir ./stage3_covid19
python stage3_covid_test.py --save_path ./stage3_covid19 --dpm-checkpoint ./stage2/model_stage2_covid_30000.pt \
--cls-checkpoint ./stage3/stage3_covid19_model_10000.pt --synthesize-number 500
```
After that, we can get synthesized sample pairs in the folder `./stage3_covid19`.

For synthesizing Image-Mask pairs with CGMH Pelvis, we run
```bash
mkdir ./stage3_cgmh
python stage3_cgmh_test.py --save_path ./stage3_cgmh --dpm-checkpoint ./stage2/model_stage2_cgmh_30000.pt \
--cls-checkpoint ./stage3/stage3_cgmh_model_10000.pt --synthesize-number 500
```
After that, we can get synthesized sample pairs in the folder `./stage3_cgmh`.

---

#### Choose high quality Image-Mask pairs (Stage IV)
Before proceeding to Stage IV, two additional things need to be done: first, train a unet using `eval.py` and save its corresponding checkpoint, and second, synthesize enough samples pairs at Stage III to facilitate the selection of high-quality sample pairs.

Then for COVID-19, we can run
```bash
mkdir ./stage4_covid19
python stage4_train.py --unet-checkpoint /path/to/unet/checkpoint --stage3-output ./stage3_covid19 \
--stage4-output ./stage4_covid19
```
And for CGMH Pelvis, we can run
```bash
mkdir ./stage4_cgmh
python stage4_train.py --unet-checkpoint /path/to/unet/checkpoint --stage3-output ./stage3_cgmh \
--stage4-output ./stage4_cgmh
```

---

### Train the validated model

For COVID-19, we can run
```bash
python eval.py --dataset=COVID19 --loss_type sigmoid_l1 --model=Unet --train_epochs=50 \
--generate_data_path ./stage4_covid19 \
--data_path /path/to/covid-chestxray-dataset/image --csv_path /path/to/covid-chestxray-dataset/metadata.csv \
```

And for CGMH Pelvis, we can run
```bash
python eval.py --dataset=CGMH --loss_type sigmoid_l1 --model=Unet --train_epochs=50 \
--generate_data_path ./stage4_cgmh \
--data_path /path/to/CGMH_PelvisSegment \
```