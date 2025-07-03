export CUDA_VISIBLE_DEVICES=0
# or use multi-gpu training when you want:
# export CUDA_VISIBLE_DEVICES=0,1

model=esm2_150m
exp=${model}
dataset=cath_4.2
name=${dataset}/${model}

python ./train.py \
    experiment=${exp} datamodule=${dataset} name=${name} \
    trainer=default logger=csv