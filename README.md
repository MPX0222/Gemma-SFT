# Gemma-SFT
> This is a script for supervised fine-tuning of Gemma models

## Steps
1. Clone the project of `https://github.com/yongzhuo/gemma-sft` and create the environment as mention.
2. Download the weights of Gemma-2b from huggingface web if needed (This script could only support running Gemma-2b in a single 4090 GPU, the usage of GPU RAM is about 15GB).
3. set the params in  `gemma_tune.py`.
4. run `gemma_tune.py`.

## Thanks for
1. https://github.com/yongzhuo/gemma-sft
