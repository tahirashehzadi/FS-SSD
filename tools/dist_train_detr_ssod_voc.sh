#!/usr/bin/env bash
TYPE=$1
FOLD=1
PERCENT=10
GPUS=2

rangeStart=29620
rangeEnd=29630

PORT=0
# 判断当前端口是否被占用，没被占用返回0，反之1
function Listening {
   TCPListeningnum=`netstat -an | grep ":$1 " | awk '$1 == "tcp" && $NF == "LISTEN" {print $0}' | wc -l`
   UDPListeningnum=`netstat -an | grep ":$1 " | awk '$1 == "udp" && $NF == "0.0.0.0:*" {print $0}' | wc -l`
   (( Listeningnum = TCPListeningnum + UDPListeningnum ))
   if [ $Listeningnum == 0 ]; then
       echo "0"
   else
       echo "1"
   fi
}

# 指定区间随机数
function random_range {
   shuf -i $1-$2 -n1
}



# 得到随机端口
function get_random_port {
   templ=0
   while [ $PORT == 0 ]; do
       temp1=`random_range $1 $2`
       if [ `Listening $temp1` == 0 ] ; then
              PORT=$temp1
       fi
   done
   echo "Using Port=$PORT"
}

# main
get_random_port ${rangeStart} ${rangeEnd};
# Start the timer
start_time=$(date +%s)
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
if [[ ${TYPE} == 'dino_detr_ssod' ]]; then
    WORK_DIR="work_dirs/2gpus_voc/40k_detr_ssod_dino_detr_r50_coco/${PERCENT}/${FOLD}/"
    mkdir -p $WORK_DIR
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port $PORT \
        $(dirname "$0")/train_detr_ssod.py configs/detr_ssod/detr_ssod_dino_detr_r50_voc_80k.py \
        --work-dir $WORK_DIR \
        --launcher pytorch \
        --cfg-options fold=${FOLD} percent=${PERCENT} ${@:5} 
fi


end_time=$(date +%s)
# Calculate and print training time
training_time=$((end_time - start_time))
echo "Training Time: $training_time seconds"

# Save training_time and final_gpu_memory in a result file
echo "Training Time: $training_time seconds" >> training_results_no_nms1.txt
