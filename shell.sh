#!/bin/bash
#
#python eval.py --dataset=COVID19 --loss_type sigmoid_l1 --model=Unet --train_epochs=50 --generate_data_path ./stage3_tau_1 \
#--num_experts=100 --buffer_path=/home/Bigdata/mtt_distillation_ckpt \
#--data_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/images/ \
#--csv_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv
#
#python eval.py --dataset=COVID19 --loss_type sigmoid_l1 --model=Unet --train_epochs=50 --generate_data_path ./stage3_tau_1 \
#--num_experts=100 --buffer_path=/home/Bigdata/mtt_distillation_ckpt \
#--data_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/images/ \
#--csv_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv

#python eval.py --dataset=COVID19 --loss_type sigmoid_l1 --model=Unet --train_epochs=50 --generate_data_path ./stage4_tau_0.5 \
#--num_experts=100 --buffer_path=/home/Bigdata/mtt_distillation_ckpt \
#--data_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/images/ \
#--csv_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv
#
#python eval.py --dataset=COVID19 --loss_type sigmoid_l1 --model=Unet --train_epochs=50 --generate_data_path ./stage4_tau_0.5 \
#--num_experts=100 --buffer_path=/home/Bigdata/mtt_distillation_ckpt \
#--data_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/images/ \
#--csv_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv
#
#python eval.py --dataset=COVID19 --loss_type sigmoid_l1 --model=Unet --train_epochs=50 --generate_data_path ./stage4_tau_0.5 \
#--num_experts=100 --buffer_path=/home/Bigdata/mtt_distillation_ckpt \
#--data_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/images/ \
#--csv_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv

python eval.py --dataset=COVID19 --loss_type sigmoid_l1 --model=Unet --train_epochs=50 --generate_data_path ./origin \
--num_experts=100 --buffer_path=/home/Bigdata/mtt_distillation_ckpt \
--data_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/images/ \
--csv_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv

python eval.py --dataset=COVID19 --loss_type sigmoid_l1 --model=Unet --train_epochs=50 --generate_data_path ./origin \
--num_experts=100 --buffer_path=/home/Bigdata/mtt_distillation_ckpt \
--data_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/images/ \
--csv_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv

python eval.py --dataset=COVID19 --loss_type sigmoid_l1 --model=Unet --train_epochs=50 --generate_data_path ./origin \
--num_experts=100 --buffer_path=/home/Bigdata/mtt_distillation_ckpt \
--data_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/images/ \
--csv_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv

#python eval.py --dataset=COVID19 --loss_type sigmoid_l1 --model=Unet --train_epochs=50 --generate_data_path ./stage4_tau_1 \
#--num_experts=100 --buffer_path=/home/Bigdata/mtt_distillation_ckpt \
#--data_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/images/ \
#--csv_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv
#
#python eval.py --dataset=COVID19 --loss_type sigmoid_l1 --model=Unet --train_epochs=50 --generate_data_path ./stage4_tau_1 \
#--num_experts=100 --buffer_path=/home/Bigdata/mtt_distillation_ckpt \
#--data_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/images/ \
#--csv_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv
#
#python eval.py --dataset=COVID19 --loss_type sigmoid_l1 --model=Unet --train_epochs=50 --generate_data_path ./stage4_tau_1 \
#--num_experts=100 --buffer_path=/home/Bigdata/mtt_distillation_ckpt \
#--data_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/images/ \
#--csv_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv

#python eval.py --dataset=COVID19 --loss_type sigmoid_l1 --model=Unet --train_epochs=50 --generate_data_path ./stage2 \
#--num_experts=100 --buffer_path=/home/Bigdata/mtt_distillation_ckpt \
#--data_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/images/ \
#--csv_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv
#
#python eval.py --dataset=COVID19 --loss_type sigmoid_l1 --model=Unet --train_epochs=50 --generate_data_path ./stage2 \
#--num_experts=100 --buffer_path=/home/Bigdata/mtt_distillation_ckpt \
#--data_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/images/ \
#--csv_path=/home/Bigdata/medical_dataset/COVID/covid-chestxray-dataset-master/metadata.csv