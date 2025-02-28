import json
from collections import Counter, defaultdict
import re
from nltk.corpus import wordnet as wn

def find_most_clear_synonym(word):
    synonyms = wn.synsets(word)
    if synonyms:
        return synonyms[0].lemmas()[0].name()
    return word

def process_semantics(semantics):
    processed_semantics = defaultdict(Counter)
    # 步骤1: 对每个关键词应用近义词分析
    for key, value_str in semantics.items():
        # 找到关键词的最明确的近义词
        synonym_key = find_most_clear_synonym(key)
        words = [word.strip() for word in value_str.split(',')]
        for word in words:
            # 分解单词和计数
            match = re.match(r"(.+?)\((\d+)\)$", word)
            if match:
                word, count = match.groups()
                processed_semantics[synonym_key][word] += int(count)
            else:
                processed_semantics[synonym_key][word] += 1

    # 步骤2: 合并近义词关键词的内容
    # 注意：由于使用了defaultdict(Counter)，步骤1中已经自动处理了合并

    # 步骤3: 对每个关键词后面的内容按出现次数进行排序
    final_semantics = {}
    for key, word_counts in processed_semantics.items():
        sorted_words = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
        final_semantics[key] = ', '.join([f"{w}({c})" if c > 1 else w for w, c in sorted_words])

    return final_semantics

# def process_semantics(semantics):
#     processed_semantics = {}
#     for key, value_str in semantics.items():
#         words = [word.strip() for word in value_str.split(',')]
#         word_counts = Counter()
#         for word in words:
#             # 分解单词和计数
#             match = re.match(r"(.+?)\((\d+)\)$", word)
#             if match:
#                 word, count = match.groups()
#                 synonym = find_most_clear_synonym(word)
#                 word_counts[synonym] += int(count)
#             else:
#                 synonym = find_most_clear_synonym(word)
#                 word_counts[synonym] += 1
#
#         sorted_words = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
#         processed_semantics[key] = ', '.join([f"{w}({c})" if c > 1 else w for w, c in sorted_words])
#
#     return processed_semantics

# 加载JSON数据
with open('/home/work/ZH/LLAMA2/deepseek-7B/ROI-Semantics/ROI_semantics_subj01_extract.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 处理数据
processed_items = []
for item in data["items"]:
    processed_item = {
        "ROI": item["ROI"],
        "semantics": process_semantics(item["semantics"])
    }
    processed_items.append(processed_item)

# 构造最终的JSON对象
final_json = {"items": processed_items}

# 保存处理后的数据到新的JSON文件
with open('/home/work/ZH/LLAMA2/deepseek-7B/ROI-Semantics/ROI_semantics_subj01_extract_NLP.json', 'w', encoding='utf-8') as f:
    json.dump(final_json, f, indent=2)
