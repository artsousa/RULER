#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

MODEL=meta-llama/Meta-Llama-3-8B
BENCHMARK=synthetic

TEMPERATURE="0.0" # greedy
TOP_P="1.0"
TOP_K="32"
# SEQ_LENGTHS=("8192" "4096" "131072" "65536" "32768" "16384" )
SEQ_LENGTHS=("8192" "16384" "32768")


source config_tasks.sh
declare -n TASKS=$BENCHMARK #${TASKS[0]} ${TASKS[1]}
if [ -z "${TASKS}" ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi

MODEL_NAME=$MODEL
MODEL_DIR=/home/main/.tmp/
TOKENIZER_PATH=/home/main/.tmp/
TOKENIZER_TYPE=hf

if [ -z "${TOKENIZER_PATH}" ]; then
        TOKENIZER_PATH=${MODEL_PATH}
        TOKENIZER_TYPE="hf"
fi

NUM_SAMPLES=32
for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
        RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
        DATA_DIR="${RESULTS_DIR}/data"; mkdir -p ${DATA_DIR}
        PRED_DIR="${RESULTS_DIR}/pred"; mkdir -p ${PRED_DIR}

        for TASK in "${TASKS[@]}"; do
                echo $RESULTS_DIR, $TASK, $MAX_SEQ_LENGTH

                python data/prepare.py \
                        --save_dir ${DATA_DIR} \
                        --benchmark ${BENCHMARK} \
                        --task ${TASK} \
                        --tokenizer_path ${MODEL_NAME} \
                        --tokenizer_type ${TOKENIZER_TYPE} \
                        --max_seq_length ${MAX_SEQ_LENGTH} \
                        --model_template_type base \
                        --num_samples ${NUM_SAMPLES} \
                        ${REMOVE_NEWLINE_TAB}

		# {DATA_DIR} {PRED_DIR} {BENCHMARK} {TASK} {MODEL_FRAMEWORK} {MODEL_PATH} {TEMPERATURE} {TOPK} {TOPP} {STOPWORDS}

		python pred/call_api.py \
            		--data_dir ${DATA_DIR} \
            		--save_dir ${PRED_DIR} \
            		--benchmark ${BENCHMARK} \
            		--task ${TASK} \
            		--server_type "hf" \
            		--model_name_or_path ${MODEL_NAME} \
            		--temperature ${TEMPERATURE} \
            		--top_k ${TOP_K} \
            		--top_p ${TOP_P} \
            		${STOP_WORDS}

        done
	
	python eval/evaluate.py \
        	--data_dir ${PRED_DIR} \
        	--benchmark ${BENCHMARK}	
	break
done
