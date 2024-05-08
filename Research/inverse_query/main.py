
import argparse
import os
import json
from vllm import LLM, SamplingParams

# https://docs.vllm.ai/en/latest/serving/distributed_serving.html#distributed-inference-and-serving
# pip install ray

# Create the parser
parser = argparse.ArgumentParser(description="Reverse query generation")

# Add arguments
parser.add_argument("--data_path", help="Raw data directory.", type=str)
parser.add_argument("--output_path", help="Output data directory.", type=str)
parser.add_argument("--model_path", help="SFT model path.", type=str)
parser.add_argument("--n_gpu", help="Number of GPUs.", type=int, default=1)
parser.add_argument("--max_new_tokens", help="Max num of new tokens.", type=int, default=512)
parser.add_argument("--batch_size", help="Batch size.", type=int, default=32)
parser.add_argument("--prompt_path", help="Generation prompt path.", type=str)
# "/home/jeeves/Yi-34B-Chat"

TEMPERATURE = 0.8
TOP_P = 0.95

if __name__ == "__main__":
    # Parse the arguments
    args = parser.parse_args()
    
    if args.n_gpu > 1:
        llm = LLM(model=args.model_path, tensor_parallel_size=args.n_gpu)
    else:
        llm = LLM(model=args.model_path)

    sampling_params = SamplingParams(
        temperature=TEMPERATURE, 
        top_p=TOP_P, 
        max_tokens=args.max_new_tokens, 
        stop=['}\n```']
    )
    
    os.makedirs(args.output_path, exist_ok=True)
    
    # Load dataset, args.data_path is a directory of jsons
    json_data = os.listdir(args.data_path)
    json_data_effective = [d for d in json_data if d.endswith('.json')]
    print(json_data_effective)
    total_data = []
    total_data_splitter = [0]
    for json_path in json_data_effective:
        json_path_abs = os.path.join(args.data_path, json_path)
        print(f"loading {json_path_abs}")
        data = json.load(open(json_path_abs, 'r'))
        total_data.extend(data)
        total_data_splitter.append(len(total_data))
    
    print(f"total # of data = {len(total_data)}")
    
    # 获取当前脚本的完整路径
    script_path = __file__

    # 获取脚本所在的目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(script_path))

    # apply prompt
    prompt_abs_path = os.path.join(script_dir, args.prompt_path)
    print(f"prompt path = {prompt_abs_path}")
    prompt_text = open(prompt_abs_path, 'r').read()
    
    filled_prompt = [prompt_text.format(passage=d) for d in total_data]
    
    # Perform inference with LLM
    batch_size = args.batch_size
    n_batches = len(filled_prompt) // batch_size + 1
    for batch_idx in range(n_batches):
        print(f"batch_idx = {batch_idx} / {n_batches}")
        batch_d = filled_prompt[batch_idx*batch_size: (batch_idx+1)*batch_size]
        outputs = llm.generate(batch_d, sampling_params)

        queries = [o.outputs[0].text for o in outputs]
        
        with open(os.path.join(args.output_path, f'queries_{batch_idx}.jsonl'), 'w') as f:
            f.write(json.dumps(queries, indent=4, ensure_ascii=False))
            
        # print(queries)
    
    
    
    
