import json
from collections import Counter, defaultdict
import re

def sort_semantic(file_path, output_json_path):
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    processed_items = []  # 用于存储处理后的每个项

    for item in data:  # 假设原始数据包含在items键下
        semantics_dict = defaultdict(list)
        removed_semantics_prefix = item['semantics'].replace("Semantics: ", "")
        keywords_list = set(re.findall(r'(\w+):', removed_semantics_prefix))

        for line in removed_semantics_prefix.split('\n'):
            for keyword in keywords_list:
                pattern = rf"{keyword}:\s*([^:\n]+)"
                matches = re.findall(pattern, line)
                for match in matches:
                    semantics_dict[keyword].extend(match.strip().split(', '))

        for keyword, contents in semantics_dict.items():
            cleaned_contents = []
            for content in contents:
                # 分割内容中的每个词，并移除任何和关键词列表中相同的词
                cleaned_content = ' '.join(word for word in content.split() if word not in keywords_list)
                cleaned_contents.append(cleaned_content)
            # 更新该关键词下的内容为处理后的内容
            semantics_dict[keyword] = cleaned_contents

        # 统计并排序
        sorted_semantics_dict = {}
        for keyword, contents in semantics_dict.items():
            content_counter = Counter(contents)
            sorted_contents = sorted(content_counter.items(), key=lambda x: (-x[1], x[0]))
            sorted_semantics_dict[keyword] = [f"{item[0]}({item[1]})" if item[1] > 1 else item[0] for item in
                                              sorted_contents]
        # 格式化内容
        formatted_semantics = {k: ', '.join(v) for k, v in sorted_semantics_dict.items()}

        # 只保留 "Location", "Action", "Object", "Color" 四项
        filtered_semantics = {k: v for k, v in formatted_semantics.items() if k in {"Location", "Action", "Object", "Color"}}

        # 构造处理后的项，保留原始的ROI值
        processed_item = {
            "ROI": item["ROI"],
            "semantics": filtered_semantics
        }
        processed_items.append(processed_item)

    # 构造最终的JSON对象，包含所有处理后的项
    final_json = {
        "items": processed_items
    }

    data = final_json["items"]
    processed_items = []
    for item in data:
        semantics = item["semantics"]  # 直接获取每个项的semantics字段

        # 根据每个关键字后面内容的数量进行排序
        sorted_semantics_keys = sorted(semantics.keys(), key=lambda k: len(semantics[k].split(', ')), reverse=True)

        # 创建一个新的排序后的semantics字典
        sorted_semantics = {key: semantics[key] for key in sorted_semantics_keys}

        # 更新item中的semantics为排序后的版本
        item["semantics"] = sorted_semantics
        processed_items.append(item)

    # 将处理后的数据保存回一个新的字典，然后保存到新的JSON文件
    final_json = {"items": processed_items}
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=2)

file_path = '/home/work/ZH/LLAMA2/deepseek-7B/ROI-Semantics/ROI_semantics_subj01.json'
output_json_path = '/home/work/ZH/LLAMA2/deepseek-7B/ROI-Semantics/ROI_semantics_subj01_extract.json'
sort_semantic(file_path, output_json_path)
