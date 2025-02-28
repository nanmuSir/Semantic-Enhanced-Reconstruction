import json
from collections import Counter, defaultdict
import re
# 定义关键词的合并规则
merge_rules = {
    "Object": ["type", "Object", "Objects", "Contents", "Type", "types", "object", "objects"],
    "Location": ["Location", "Locations", "location", "locations"],
    "Action": ["Action", "Actions", "Activity", "Accessories", "action", "actions", "activity"],
    "Color": ["Color", "Colors", "color", "colors"]
}

# 读取JSON文件
input_file_path = '/home/work/ZH/LLAMA2/deepseek-7B/ROI-Semantics/ROI_semantics_subj07_extract.json'
output_file_path = '/home/work/ZH/LLAMA2/deepseek-7B/ROI-Semantics/ROI_semantics_subj07_extract_merge.json'

with open(input_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)["items"]

processed_items = []
for item in data:
    # 使用defaultdict(Counter)来收集和计数合并后的内容
    new_semantics = defaultdict(Counter)

    # 遍历所有关键词，包括目标关键词和需要合并的关键词
    for target, synonyms in merge_rules.items():
        # 首先处理目标关键词自身的内容（如果有）
        if target in item["semantics"]:
            target_contents = item["semantics"][target].split(', ')
            for content in target_contents:
                match = re.match(r"(.+?)\((\d+)\)$", content)
                if match:
                    word, count = match.groups()
                    new_semantics[target][word.lower()] += int(count)
                else:
                    new_semantics[target][content.lower()] += 1

        # 然后处理和合并其它关键词的内容
        for synonym in synonyms:
            if synonym in item["semantics"] and synonym != target:
                contents = item["semantics"][synonym].split(', ')
                for content in contents:
                    match = re.match(r"(.+?)\((\d+)\)$", content)
                    if match:
                        word, count = match.groups()
                        new_semantics[target][word.lower()] += int(count)
                    else:
                        new_semantics[target][content.lower()] += 1
                # 删除被合并的关键词
                del item["semantics"][synonym]

    # 将合并和计数后的内容转换回字符串格式，并赋值给item["semantics"]
    for key, counter in new_semantics.items():
        sorted_contents = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        item["semantics"][key] = ', '.join([f"{word}({count})" if count > 1 else word for word, count in sorted_contents])

    processed_items.append(item)

# 保存处理后的数据到新的JSON文件
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump({"items": processed_items}, f, indent=2)
