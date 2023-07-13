"""
step-by-step的train,eval,chain

todo:
    1. Finetune by checkpoint.
    2. main code framework
"""


TEMP_DIR = "./step-by-step/temp"
RES_DIR = "./step-by-step/"
DATASET_LIST = ["AQuA", "AddSub", "MultiArith", "SingleEq", "gsm8k", "SVAMP"]
ADAPTERS = ["kvmlora", "lora"]

from concurrent.futures import ProcessPoolExecutor
import queue
import subprocess
from itertools import product


def step_by_step(adapter_name: str, gpuid: int):
    print("begin step-by-step experiment")
    first = True
    for dataset in DATASET_LIST:
        if first:
            # finetune
            command = f"CUDA_VISIBLE_DEVICES={gpuid} python finetune.py \
                --base_model yahma/llama-7b-hf \
                --datapath  {TODO}\
                --output_dir {TEMP_DIR}/weight-after-{dataset} \
                --base_model 'yahma/llama-7b-hf' \
                --batch_size 16 \
                --micro_batch_size 16 \
                --num_epochs 3 \
                --learning_rate 3e-4 \
                --cutoff_len 256 \
                --val_set_size 120 \
                --adapter_name ${adapter_name}"
        # evaluate
        pass
    pass


tasks = ADAPTERS

gpus = [0, 1, 2, 3, 4, 5, 6, 7]
# gpus = [3, 4, 5, 6]
# gpus = [
#     4,
# ]
tasks_queue = queue.Queue()
gpu_queue = queue.Queue()
num_processes = min(
    len(tasks), len(gpus)
)  # number of processes to run in parallel


def main():
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(step_by_step, tasks_queue.get(), gpu_queue.get())
            for i in range(num_processes)
        ]
        for future in futures:
            gpu_id = future.result()
            gpu_queue.put(gpu_id)
            if tasks_queue.qsize() > 0:
                futures.append(
                    executor.submit(
                        step_by_step, tasks_queue.get(), gpu_queue.get()
                    )
                )


if __name__ == "__main__":
    main()
