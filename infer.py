import sys, json
from typing import Optional
import fire
import torch

from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Response:
"""  # noqa: E501


def evaluate(
    instruction,
    tokenizer=None,
    model=None,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()


def main(
    lora_weights: str,
    base_model: Optional[str] = "decapoda-research/llama-7b-hf",
):
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    match device:
        case "cuda":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
                torch_dtype=torch.float32,
            )
        case _:
            raise NotImplementedError
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.half()  # seems to fix bugs for some users.

    model.eval()
    ress = []
    for instruction in tqdm(
        [
            "Tell me about alpacas.",
            "Tell me about the president of Mexico in 2019.",
            "Tell me about the king of France in 2019.",
            "List all Canadian provinces in alphabetical order.",
            "Write a Python program that prints the first 10 Fibonacci numbers.",
            "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
            "Tell me five words that rhyme with 'shock'.",
            "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
            "Count up from 1 to 500.",
            "Julia played tag with 16 kids on monday. If she played tag with 12 more kids on monday than on tuesday. How many kids did she play with on tuesday?",
            "Julia played tag with 19 kids on monday. She played tag with 18 kids on tuesday. She spent a total of 38 hours to play tag on both days. How many more kids did she play with on monday than on tuesday?",
            "Robin's hair was 19 inches long. If he grew 18 more inches. How long is his hair now?",
            " Haley was planting vegetables in her garden. She started with 56 seeds and planted 35 of them in the big garden and in each of her small gardens put 3 seeds each. How many small gardens did Haley have? ",
        ]
    ):
        res = evaluate(instruction, tokenizer, model)
        obj = {"instruction": instruction, "input": "", "output": res}
        ress.append(obj)
    with open(
        f"./evaluate_res/evaluate_{'_'.join(lora_weights.split('/')[-2:])}.json",
        "w",
    ) as f:
        json.dump(ress, f)


if __name__ == "__main__":
    fire.Fire(main)
