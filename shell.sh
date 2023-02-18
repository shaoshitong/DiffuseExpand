#!/bin/bash

python eval.py --dataset=CGMH --loss_type sigmoid_l1 --model=AttnUnet --train_epochs=50 --generate_data_path /home/Bigdata/medical_dataset/output/CGMH/stage4_tau_0.33_scale_1.0 \
--num_experts=100 --buffer_path=/home/Bigdata/mtt_distillation_ckpt \
--data_path=/home/Bigdata/medical_dataset/CGMH_PelvisSegment \
--csv_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv

python eval.py --dataset=CGMH --loss_type sigmoid_l1 --model=AttnUnet --train_epochs=50 --generate_data_path /home/Bigdata/medical_dataset/output/CGMH/stage4_tau_0.33_scale_1.0 \
--num_experts=100 --buffer_path=/home/Bigdata/mtt_distillation_ckpt \
--data_path=/home/Bigdata/medical_dataset/CGMH_PelvisSegment \
--csv_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv

python eval.py --dataset=CGMH --loss_type sigmoid_l1 --model=AttnUnet --train_epochs=50 --generate_data_path /home/Bigdata/medical_dataset/output/CGMH/stage4_tau_0.33_scale_1.0 \
--num_experts=100 --buffer_path=/home/Bigdata/mtt_distillation_ckpt \
--data_path=/home/Bigdata/medical_dataset/CGMH_PelvisSegment \
--csv_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv

#
#python eval.py --dataset=CGMH --loss_type sigmoid_l1 --model=Unet --train_epochs=50 --generate_data_path /home/Bigdata/medical_dataset/output/CGMH/stage4_tau_0.5_scale_1.0 \
#--num_experts=100 --buffer_path=/home/Bigdata/mtt_distillation_ckpt \
#--data_path=/home/Bigdata/medical_dataset/CGMH_PelvisSegment \
#--csv_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv
#
#python eval.py --dataset=CGMH --loss_type sigmoid_l1 --model=Unet --train_epochs=50 --generate_data_path /home/Bigdata/medical_dataset/output/CGMH/stage4_tau_0.5_scale_1.0 \
#--num_experts=100 --buffer_path=/home/Bigdata/mtt_distillation_ckpt \
#--data_path=/home/Bigdata/medical_dataset/CGMH_PelvisSegment \
#--csv_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv
#
#python eval.py --dataset=CGMH --loss_type sigmoid_l1 --model=Unet --train_epochs=50 --generate_data_path /home/Bigdata/medical_dataset/output/CGMH/stage4_tau_0.5_scale_1.0 \
#--num_experts=100 --buffer_path=/home/Bigdata/mtt_distillation_ckpt \
#--data_path=/home/Bigdata/medical_dataset/CGMH_PelvisSegment \
#--csv_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv
#
#
#python eval.py --dataset=CGMH --loss_type sigmoid_l1 --model=Unet --train_epochs=50 --generate_data_path /home/Bigdata/medical_dataset/output/CGMH/stage4_tau_0.33_scale_1.0 \
#--num_experts=100 --buffer_path=/home/Bigdata/mtt_distillation_ckpt \
#--data_path=/home/Bigdata/medical_dataset/CGMH_PelvisSegment \
#--csv_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv
#
#python eval.py --dataset=CGMH --loss_type sigmoid_l1 --model=Unet --train_epochs=50 --generate_data_path /home/Bigdata/medical_dataset/output/CGMH/stage4_tau_0.33_scale_1.0 \
#--num_experts=100 --buffer_path=/home/Bigdata/mtt_distillation_ckpt \
#--data_path=/home/Bigdata/medical_dataset/CGMH_PelvisSegment \
#--csv_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv
#
#python eval.py --dataset=CGMH --loss_type sigmoid_l1 --model=Unet --train_epochs=50 --generate_data_path /home/Bigdata/medical_dataset/output/CGMH/stage4_tau_0.33_scale_1.0 \
#--num_experts=100 --buffer_path=/home/Bigdata/mtt_distillation_ckpt \
#--data_path=/home/Bigdata/medical_dataset/CGMH_PelvisSegment \
#--csv_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv
