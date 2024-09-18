# Use a pipeline as a high-level helper
from transformers import pipeline
import os

# 设置缓存目录
os.environ['TRANSFORMERS_CACHE'] = '~/jyuzh/Practice/TangChuang/·model/.cache'
pipe = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", cache_dir='~/jyuzh/Practice/TangChuang/·model/.cache')    

# 多条语句
sentences = [
    "Barack Obama was born in Kenya",
    "The earth is flat",
    "Python is a programming language"
]

# 处理多条语句
results = pipe(
    sentences,
    candidate_labels=["fact", "fiction"],
    hypothesis_template="This statement is {}."
)

# 打印结果
for result in results:
    print(result)