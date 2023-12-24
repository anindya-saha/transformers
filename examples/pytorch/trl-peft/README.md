## Examples of using peft with trl to finetune 8-bit models with Low Rank Adaption (LoRA)
Ref: https://huggingface.co/docs/trl/v0.7.4/en/lora_tuning_peft

### Create an Accelerator Config for Multi GPU training
```bash
$:~/transformers-asaha/examples/pytorch/trl-peft$ accelerate config
[2023-12-23 01:40:02,552] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
In which compute environment are you running?
This machine                                                                                                                       
Which type of machine are you using?                                                                                               
multi-GPU                                                                                                                          
How many different machines will you use (use more than 1 for multi-node training)? [1]:                                           
Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]: NO  
Do you wish to optimize your script with torch dynamo?[yes/NO]:NO                                                                  
Do you want to use DeepSpeed? [yes/NO]: NO                                                                                         
Do you want to use FullyShardedDataParallel? [yes/NO]: NO                                                                          
Do you want to use Megatron-LM ? [yes/NO]: NO                                                                                      
How many GPU(s) should be used for distributed training? [1]:8
What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:all
-Do you wish to use FP16 or BF16 (mixed precision)?
bf16   
-------------------------------------------------------------------------------------------------
accelerate configuration saved at /home/anindya/.cache/huggingface/accelerate/default_config.yaml                                  
(starcoder-tune) anindya@devanindya-a100-40gb-8gpu:~/transformers-asaha/examples/pytorch/trl-peft$ 
```

```bash
cp /home/anindya/.cache/huggingface/accelerate/default_config.yaml accelerate_config.yaml
```

### Launch Multi GPU training in one node
```bash
TRANSFORMERS_VERBOSITY=info accelerate launch --config_file accelerate_config.yaml sft.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset_name timdettmers/openassistant-guanaco \
    --do_train \
    --do_eval \
    --output_dir ./tmp/llama2-guanaco \
    --load_in_4bit \
    --use_peft \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --bf16 \
    --num_train_epochs 10 \
    --report_to none \
    --ddp_find_unused_parameters False \
    --push_to_hub True \
    --hub_private_repo True \
    --hub_model_id asaha-cdcp/llama2-guanaco \
    --hub_token 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
```
The training is configured to resume from the last checkpoint. To avoid this behavior, change the `--output_dir` or 
add `--overwrite_output_dir` to train from scratch.

Additional optimizer parameters:

```bash
    --optim paged_adamw_32bit \
    --learning_rate 2e-4 \
    --max_grad_norm 0.3 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type linear \
```

### Logging Metrics to Weights & Biases

Set the following env variables:
```bash
export WANDB_PROJECT=TRL-Peft
export WANDB_API_KEY=9b0301aa37b1bd96055367dd584e2d969444146d
export WANDB_RUN_GROUP=Multi-Gpu-Experiments
```

Also, set the following in the TrainingArguments:
```bash
--report_to wandb
```

**If you don't want your script to sync to the wandb cloud**
```bash
os.environ["WANDB_MODE"] = "offline"
```

### Check GPU Utilization

```bash
$ watch nvidia-smi

Sat Dec 23 04:36:03 2023
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          Off | 00000000:00:04.0 Off |                    0 |
| N/A   75C    P0             375W / 400W |  39661MiB / 40960MiB |     94%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA A100-SXM4-40GB          Off | 00000000:00:05.0 Off |                    0 |
| N/A   61C    P0             341W / 400W |  39978MiB / 40960MiB |     93%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   2  NVIDIA A100-SXM4-40GB          Off | 00000000:00:06.0 Off |                    0 |
| N/A   68C    P0             398W / 400W |  39674MiB / 40960MiB |     94%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   3  NVIDIA A100-SXM4-40GB          Off | 00000000:00:07.0 Off |                    0 |
| N/A   76C    P0             390W / 400W |  39674MiB / 40960MiB |     94%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   4  NVIDIA A100-SXM4-40GB          Off | 00000000:80:00.0 Off |                    0 |
| N/A   67C    P0             384W / 400W |  39646MiB / 40960MiB |    100%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   5  NVIDIA A100-SXM4-40GB          Off | 00000000:80:01.0 Off |                    0 |
| N/A   80C    P0             392W / 400W |  39608MiB / 40960MiB |     94%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   6  NVIDIA A100-SXM4-40GB          Off | 00000000:80:02.0 Off |                    0 |
| N/A   65C    P0             326W / 400W |  39646MiB / 40960MiB |    100%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   7  NVIDIA A100-SXM4-40GB          Off | 00000000:80:03.0 Off |                    0 |
| N/A   72C    P0             345W / 400W |  39530MiB / 40960MiB |    100%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      2358      G   /usr/lib/xorg/Xorg                           95MiB |
|    0   N/A  N/A      2665      G   /usr/bin/gnome-shell                         12MiB |
|    0   N/A  N/A     20077      C   .../anindya/starcoder-tune/bin/python3    39526MiB |
|    1   N/A  N/A      2358      G   /usr/lib/xorg/Xorg                            4MiB |
|    1   N/A  N/A     20078      C   .../anindya/starcoder-tune/bin/python3    39952MiB |
|    2   N/A  N/A      2358      G   /usr/lib/xorg/Xorg                            4MiB |
|    2   N/A  N/A     20079      C   .../anindya/starcoder-tune/bin/python3    39648MiB |
|    3   N/A  N/A      2358      G   /usr/lib/xorg/Xorg                            4MiB |
|    3   N/A  N/A     20080      C   .../anindya/starcoder-tune/bin/python3    39648MiB |
|    4   N/A  N/A      2358      G   /usr/lib/xorg/Xorg                            4MiB |
|    4   N/A  N/A     20081      C   .../anindya/starcoder-tune/bin/python3    39620MiB |
|    5   N/A  N/A      2358      G   /usr/lib/xorg/Xorg                            4MiB |
|    5   N/A  N/A     20082      C   .../anindya/starcoder-tune/bin/python3    39582MiB |
|    6   N/A  N/A      2358      G   /usr/lib/xorg/Xorg                            4MiB |
|    6   N/A  N/A     20083      C   .../anindya/starcoder-tune/bin/python3    39620MiB |
|    7   N/A  N/A      2358      G   /usr/lib/xorg/Xorg                            4MiB |
|    7   N/A  N/A     20084      C   .../anindya/starcoder-tune/bin/python3    39504MiB |
+---------------------------------------------------------------------------------------+
```

### Test Fine Tuned Model

```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

peft_model_id = '/home/anindya/transformers-asaha/examples/pytorch/trl-peft/tmp/llama2-guanaco/final'
hub_token = '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'

peft_config = PeftConfig.from_pretrained(peft_model_id, token=hub_token)
base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, token=hub_token)
merged_model = PeftModel.from_pretrained(base_model, peft_model_id, token=hub_token)

tokenizer = AutoTokenizer.from_pretrained(
    peft_model_id, 
    token=hub_token
)

prompt = "Who is Leonardo Da Vinci?"
pipe = pipeline(task="text-generation", model=merged_model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
```

### Debugging the script
Start your script without `accelerate` as a normal python program and read the `args` from a file. 

Create a file `sft.args` and put all arguments where each line corresponds to a single argument and 
its value, with the flags set as needed.

Modify the `parser.parse_args_into_dataclasses(look_for_args_file=True)` to have the flag 
`look_for_args_file=True` so that it looks for a file `sft.args` instead of `sys.argv`