import random

random.seed(42)

# Load all lines from the file
with open('parquet_index.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Randomly sample 1000 lines
sampled_lines = random.sample(lines, 2000)

# Save the sampled lines to a new file
with open('sampled_parquet_index.txt', 'w', encoding='utf-8') as file:
    file.writelines(sampled_lines)
