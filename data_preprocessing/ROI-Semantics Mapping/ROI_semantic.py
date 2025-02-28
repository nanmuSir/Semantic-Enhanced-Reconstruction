import json

# merge ROI and semantics/description
def ROI_semantic(infile1, infile2, outfile) :
    with open(infile1, "r") as f1:
        json_data1 = json.load(f1)

    with open(infile2, "r") as f2:
        json_data2 = json.load(f2)

    merged_json = {}
    i = 0
    # 遍历第一个 JSON 文件的 item，同时获取第二个 JSON 文件中的 "semantics"
    for (key1, value1), item2 in zip(json_data1.items(), json_data2):
        # 将第一个 JSON 文件的 value 作为新 JSON 文件的 key
        # keys = [int(k) for k in value1.split(",")]
        keys = [int(k) for idx, k in enumerate(value1.split(",")) if idx % 3 == 0]

        # 获取第二个 JSON 文件中对应的 "semantics"
        semantics = item2.get("semantics", "")
        # 对每个元素进行处理，替换其中的特殊字符
        processed_semantics = [s.replace("\n*", "") for s in semantics]
        # 如果后续需要合并成字符串
        semantics = ''.join(processed_semantics)
        semantics = semantics.replace("\n*", "")
        # 遍历新的 keys
        for new_key in keys:
            # 如果新 JSON 文件中已存在相同的 key，就合并两个 key
            if new_key in merged_json:
                merged_json[new_key]["semantics"] += "\n" + semantics
            else:
                # 否则，创建新的条目
                merged_json[new_key] = {"ROI": new_key, "semantics": semantics}

        i = i + 1
        if i%1000==0 :
            print(infile1,'    ',i)

    with open(outfile, "w") as output_json:
        json.dump(list(merged_json.values()), output_json, indent=2)

    print(f"Merged result saved to: {outfile}")

infile1 = "/home/data/ZH/NSD/nsddata_betas/ppdata/subj07/2%smooth/active-ROI_value.json"
infile2 = "/home/work/ZH/LLAMA2/deepseek-7B/semantic_result/merged_semantics_subj07.json"
outfile = "/home/work/ZH/LLAMA2/deepseek-7B/ROI-Semantics/ROI_semantics_subj07.json"
ROI_semantic(infile1, infile2, outfile)

