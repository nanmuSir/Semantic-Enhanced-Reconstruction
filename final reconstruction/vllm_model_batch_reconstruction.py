# vllm_semantic_extractor.py
import torch
from vllm import LLM, SamplingParams
import json
import os
from tqdm import tqdm
import re
import fire
import torch.distributed as dist
import atexit

class SceneDescGenerator:
    def __init__(self,
                 model_path: str,
                 input_file: str,
                 output_dir: str,
                 batch_size: int = 16,
                 max_tokens: int = 1024,
                 temperature: float = 0.1,
                 chunk_size: int = 2000):

        self.model_path = model_path
        self.input_file = input_file
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.temperature = temperature
        self.max_tokens = max_tokens

        os.makedirs(output_dir, exist_ok=True)

        # 初始化模型
        self.llm = LLM(
            model=model_path,
            dtype="float16",
            max_model_len=4096,
            tensor_parallel_size=1,
            trust_remote_code=True,
            enable_prefix_caching=True
            # gpu_memory_utilization=0.9
        )

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens,
            stop_token_ids=[151329, 151336, 151338]  # DeepSeek停止符
        )

        # 注册清理函数
        atexit.register(self.cleanup)

    def cleanup(self):
        """清理函数，确保 NCCL 资源被释放"""
        if dist.is_available() and dist.is_initialized():
            print("Destroying process group...")
            dist.destroy_process_group()
        if torch.cuda.is_available():
            print("Clearing GPU cache...")
            torch.cuda.empty_cache()

    def _extract_keywords(self, field: str) -> str:
        """从带权重的语义字段中提取纯关键词"""
        return ', '.join([term.split('(')[0].strip() for term in field.split(',')])

    def build_prompts(self, items: list) -> list:
        """构造生成场景描述的prompt"""
        prompts = []
        for item in items:
            desc = item["description"]
            semantics = item["semantics"]

            # 提取并格式化语义关键词
            locations = self._extract_keywords(semantics["Location"])
            actions = self._extract_keywords(semantics["Action"])
            colors = self._extract_keywords(semantics["Color"])
            objects = self._extract_keywords(semantics["Object"])

            system_msg = """你是一个场景描述生成专家。请严格遵循以下步骤：
1. 仔细分析原始描述的核心要素和氛围
2. 从语义关键词中选择与原始场景最相关的词语
3. 如果相关语义词过少，可以额外使用1个原始描述中的语义词
"""
            user_msg = f"""原始描述：{desc}
可用语义词：
- 地点：{locations}
- 动作：{actions}
- 颜色：{colors}
- 物体：{objects}

请生成贴合原始场景的自然流畅的新描述,只给出答案，禁止添加任何解释或思考内容! ：
<think>\n"""

            prompts.append(f"<s>[INST] {system_msg}\n{user_msg} [/INST]\n")
        return prompts

    @staticmethod
    def clean_description(text: str) -> str:
        """后处理生成结果"""
        text = re.sub(r'\s+', ' ', text)  # 合并空格
        text = re.sub(r'\[.*?\]', '', text)  # 去除可能存在的标记
        text = text.replace('"', '').strip()  # 去除引号

        # 确保以句号结尾
        if text and text[-1] not in {'.', '!', '?'}:
            text += '.'
        return text

    def process(self):
        """主处理流程"""
        # 加载数据
        with open(self.input_file) as f:
            data_dict = json.load(f)
            # print('len of data:',len(data_dict))

        # 转换为列表并过滤无效数据
        data_list = [
            {"image_id": k, "semantics": v["semantics"], "description": v["description"]}
            for k, v in data_dict.items()
            if "semantics" in v and "description" in v
        ]

        results = []
        progress = tqdm(total=len(data_list), desc="Generating")

        # 批量处理
        for i in range(0, len(data_list), self.batch_size):
            batch = data_list[i:i + self.batch_size]

            # 生成prompts
            prompts = self.build_prompts(batch)

            # 模型推理
            outputs = self.llm.generate(prompts, self.sampling_params)

            # 处理结果
            for item, output in zip(batch, outputs):
                generated = self.clean_description(output.outputs[0].text)

                results.append({
                    "image_id": item["image_id"],
                    "original_description": item["description"],
                    "semantics": item["semantics"],
                    "generated_description": generated
                })

                # 分块保存
                if len(results) >= self.chunk_size:
                    self.save_chunk(results)
                    results = []

                progress.update(1)

        # 保存剩余数据
        if results:
            self.save_chunk(results)

        progress.close()

    def save_chunk(self, data: list):
        """保存分块数据"""
        # chunk_id = len(os.listdir(self.output_dir)) + 1
        output_path = os.path.join(self.output_dir, f"reconstruction_subj07.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # print(f"Saved chunk {chunk_id} with {len(data)} items")

def main(
        model_path: str = "/home/work/ZH/LLAMA2/deepseek-7B/model",
        input_file: str = "/home/work/ZH/StableDiffusionReconstruction-main/semantic_cluster/test_semantic/test_cocoid_semantic_desp_subj07.json",
        output_dir: str = "/home/work/ZH/LLAMA2/deepseek-7B",
        batch_size: int = 16,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        chunk_size: int = 2000
):
    generator = SceneDescGenerator(
        model_path=model_path,
        input_file=input_file,
        output_dir=output_dir,
        batch_size=batch_size,
        max_tokens=max_tokens,
        temperature=temperature,
        chunk_size=chunk_size
    )
    generator.process()

if __name__ == "__main__":
    # fire.Fire(main)
    main()
