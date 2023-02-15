#!/bin/bash

python eval.py --dataset=COVID19 --loss_type sigmoid_l1 --model=Unet --train_epochs=50 --generate_data_path ./stage4_tau_1 \
--num_experts=100 --buffer_path=/home/Bigdata/mtt_distillation_ckpt \
--data_path=./origin --ratio 0.7 \
--csv_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv

python eval.py --dataset=COVID19 --loss_type sigmoid_l1 --model=Unet --train_epochs=50 --generate_data_path ./stage4_tau_1 \
--num_experts=100 --buffer_path=/home/Bigdata/mtt_distillation_ckpt \
--data_path=./origin --ratio 0.7 \
--csv_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv


#python eval.py --dataset=CGMH --loss_type sigmoid_l1 --model=Unet --train_epochs=50 --generate_data_path /home/Bigdata/medical_dataset/output/CGMH/stage3_tau_1.0_scale_1.0 \
#--num_experts=100 --buffer_path=/home/Bigdata/mtt_distillation_ckpt \
#--data_path=/home/Bigdata/medical_dataset/CGMH_PelvisSegment \
#--csv_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv
#
#python eval.py --dataset=CGMH --loss_type sigmoid_l1 --model=Unet --train_epochs=50 --generate_data_path /home/Bigdata/medical_dataset/output/CGMH/stage3_tau_1.0_scale_1.0 \
#--num_experts=100 --buffer_path=/home/Bigdata/mtt_distillation_ckpt \
#--data_path=/home/Bigdata/medical_dataset/CGMH_PelvisSegment \
#--csv_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv
#
#python eval.py --dataset=CGMH --loss_type sigmoid_l1 --model=Unet --train_epochs=50 --generate_data_path /home/Bigdata/medical_dataset/output/CGMH/stage3_tau_1.0_scale_1.0 \
#--num_experts=100 --buffer_path=/home/Bigdata/mtt_distillation_ckpt \
#--data_path=/home/Bigdata/medical_dataset/CGMH_PelvisSegment \
#--csv_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv
