# Load model directly
from transformers import AutoModel

model = AutoModel.from_pretrained("bartowski/DeepSeek-R1-Distill-Qwen-14B-exl2")
