# data.py
import glob
import unicodedata
from typing import List, Dict
from config import all_letters, n_letters, data_path
import torch

def unicode_to_ascii(s: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in all_letters
    )

def read_lines(filename: str) -> List[str]:
    with open(filename, encoding='utf-8') as f:
        return [unicode_to_ascii(line.strip()) for line in f]

def find_files(path: str) -> List[str]:
    return glob.glob(path)

# Load all category data
category_lines: Dict[str, List[str]] = {}
all_categories: List[str] = []

for filename in find_files(data_path):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    category_lines[category] = read_lines(filename)

n_categories = len(all_categories)

def letter_to_index(letter: str) -> int:
    return all_letters.find(letter)

def line_to_tensor(line: str) -> torch.Tensor:
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor
# data.py
import glob
import unicodedata
from typing import List, Dict
from config import all_letters, n_letters, data_path
import torch

def unicode_to_ascii(s: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in all_letters
    )

def read_lines(filename: str) -> List[str]:
    with open(filename, encoding='utf-8') as f:
        return [unicode_to_ascii(line.strip()) for line in f]

def find_files(path: str) -> List[str]:
    return glob.glob(path)

# Load all category data
category_lines: Dict[str, List[str]] = {}
all_categories: List[str] = []

for filename in find_files(data_path):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    category_lines[category] = read_lines(filename)

n_categories = len(all_categories)

def letter_to_index(letter: str) -> int:
    return all_letters.find(letter)

def line_to_tensor(line: str) -> torch.Tensor:
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor
