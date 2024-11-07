from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import torch

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",
                                             torch_dtype=torch.bfloat16,
                                             device_map=0,
                                             )     

eos_token_id = tokenizer.eos_token_id

text = """You are given an input math problem and its solution. Your task is to abstract the problem and solution into a template, as shown in the example below.

### Example
Input:
Question: When Sophie watches her nephew, she gets out a variety of toys for him. The bag of building blocks has 31 blocks in it. The bin of stuffed animals has 8 stuffed animals inside. The tower of stacking rings has 9 multicolored rings on it. Sophie recently bought a tube of bouncy balls, bringing her total number of toys for her nephew up to 62. How many bouncy balls came in the tube?
Answer: Sophie has 31 + 8 + 9 + T = 62 toys. Thus, T = 62 - (31 + 8 + 9) = <<62 - (31 + 8 + 9)=14>>14. #### 14

Output:
Question: When {name} watches her {family}, she gets out a variety of toys for him. The bag of building blocks has {x} blocks in it. The bin of stuffed animals has {y} stuffed animals inside. The tower of stacking rings has {z} multicolored rings on it. {name} recently bought a tube of bouncy balls, bringing her total number of toys for her {family} up to {total}. How many bouncy balls came in the tube? 
Variables:
- name = sample(names)
- family = sample(["nephew", "cousin", "brother"])
- x = range(5, 100)
- y = range(5, 100)
- z = range(5, 100)
- total = range(100, 500)
- ans = range(85, 200)
Conditions:
- x + y + z + ans == total

Answer:
Let T be the number of bouncy balls in the tube. After buying the tube of balls, {name} has {x} + {y} + {z} + T = { x + y + z } + T = {total} toys for her {family}. Thus, T = {total} - { x + y + z } = <<{total}-{ x + y + z }={ans}>>{ans} bouncy balls came in the tube.

### New Problem
Input:
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72

Output:
"""


generation_kwargs = {
    "max_new_tokens": 1024, 
    "min_length": -1,
    "do_sample": True,
}

inputs = tokenizer(text, return_tensors="pt").to(model.device)
print("Generating")
with torch.no_grad():
    outputs = model.generate(**inputs, **generation_kwargs, pad_token_id=tokenizer.eos_token_id)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)