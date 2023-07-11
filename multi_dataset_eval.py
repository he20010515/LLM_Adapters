from concurrent.futures import ProcessPoolExecutor
import queue
import subprocess
from itertools import product


def evaluate(args, gpu):
    dataset, adaptername = args
    print("*******dataset:", dataset, "adaptername:", adaptername)

    command = f"CUDA_VISIBLE_DEVICES={gpu} python evaluate.py \
               --model LLaMA-7B \
               --adapter {adaptername} \
               --dataset {dataset} \
               --base_model 'yahma/llama-7b-hf' \
               --lora_weights './trained_models/yahmallama-{adaptername}'"

    result = subprocess.run(
        command, shell=True, text=True, capture_output=False
    )
    print(
        f"Evaluation results for dataset {dataset} on GPU {gpu}:\n{result.stdout}"
    )
    return gpu


datasets = ["AQuA", "AddSub", "MultiArith", "SingleEq", "gsm8k", "SVAMP"]
# adapternames = ["lora", "kvlora", "prototypelora", "kvmlora"]

# datasets = ["AQuA"]
adapternames = ["kvmlora"]

tasks = list(product(datasets, adapternames))

gpus = [0, 1, 2, 3, 4, 5, 6, 7]
# gpus = [3, 4, 5, 6]
# gpus = [
#     4,
# ]
tasks_queue = queue.Queue()
gpu_queue = queue.Queue()

for gpu in gpus:
    gpu_queue.put(gpu)
for task in tasks:
    tasks_queue.put(task)

num_processes = min(
    len(tasks), len(gpus)
)  # number of processes to run in parallel

with ProcessPoolExecutor(max_workers=num_processes) as executor:
    futures = [
        executor.submit(evaluate, tasks_queue.get(), gpu_queue.get())
        for i in range(num_processes)
    ]
    for future in futures:
        gpu_id = future.result()
        gpu_queue.put(gpu_id)
        if tasks_queue.qsize() > 0:
            futures.append(
                executor.submit(evaluate, tasks_queue.get(), gpu_queue.get())
            )
