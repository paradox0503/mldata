import re
import json
from datasets import load_dataset
from tqdm import tqdm
import os


def find_question(text, start_index):
    # 定义匹配包含 "No or Yes?" 或 "Yes or No?" 的问句的正则表达式
    special_question_pattern = r"\?(.*?(No or Yes\?|Yes or No\?))"
    question_match = re.search(special_question_pattern, text[start_index:], re.DOTALL)
    
    if question_match:
        # 如果匹配到包含特殊短语的问句，直接返回完整问句
        question_end_index = start_index + question_match.end()
        question_text = text[start_index:question_end_index].strip()
        return question_text, question_end_index

    # 如果没有匹配到特殊短语，按照原有逻辑解析问句
    question_match = re.search(r"\?", text[start_index:])
    if question_match:
        question_end_index = start_index + question_match.end()

        # 向前搜索换行符，从而找到问题的开头 headline（也可能没有换行）
        before_question_text = text[:question_end_index]

        # 找到上一行的起点（即 headline 的起点）
        lines = before_question_text.strip().splitlines()

        # 组合所有行，最后一行为真正问句，前面的视为上下文 headline
        if len(lines) >= 2:
            question_text = " ".join(lines[-2:])  # headline + 问句
        else:
            question_text = lines[-1]

        return question_text.strip(), question_end_index

    return None, None



# 匹配答案
def find_answer(text, start_index):

    answer_text = text[start_index:]
    # 匹配 Options 格式答案
    options_pattern = r"Options:\s*((?:-?\s*(Yes|No)\s*)+)"
    options_match = re.search(options_pattern, answer_text, re.IGNORECASE)
    if options_match:
        options_text = options_match.group(1)
        # 提取所有的 Yes 或 No
        valid_answers = re.findall(r"(Yes|No)", options_text, re.IGNORECASE)
        if valid_answers:
            last_answer = valid_answers[-1].strip()
            return "Yes" if last_answer.lower().startswith("y") else "No"

    # 匹配其他格式的答案
    answer_patterns = [
        r"No\s+or\s+Yes\?\s*(Yes|No)",
        r"\b(Yes|No|Y|N)\b",
    ]
    for pattern in answer_patterns:
        answer_match = re.search(pattern, answer_text)
        if answer_match:
            # 检查是否有有效的答案组
            valid_answer = None
            for group in answer_match.groups():
                if group and group.strip().lower() in ['yes', 'no', 'y', 'n']:
                    valid_answer = group.strip()
                    break
            if valid_answer:
                return "Yes" if valid_answer.lower().startswith("y") else "No"

    return None


def parse_qa_pairs(text):
    processed_matches = []
    start_index = 0
    while True:
        question, question_end_index = find_question(text, start_index)
        if question is None or start_index >= len(text):
            break
        answer = find_answer(text, question_end_index)
        processed_matches.append((question, answer))
        start_index = question_end_index

    return processed_matches


# 结构化数据生成
def generate_structured_data(dataset):
    structured_data = []
    error_log = []

    for entry in tqdm(dataset, desc="处理进度"):
        try:
            input_text = entry.get("input", "")
            if not input_text:
                error_log.append({"id": entry["id"], "reason": "空输入"})
                continue

            qa_pairs = parse_qa_pairs(input_text)
            if not qa_pairs:
                error_log.append({"id": entry["id"], "reason": "无有效问答对"})
                continue

            # 生成数据
            for idx, (q, a) in enumerate(qa_pairs, 1):
                structured_data.append({
                    "id": f"headline-{entry['id']}-{idx}",
                    "Question": q,
                    "Answer": a
                })

        except Exception as e:
            error_log.append({"id": entry["id"], "reason": str(e)})

    # 保存错误日志
    with open("error_log.json", "w") as f:
        json.dump(error_log, f)

    return structured_data



# 加载数据集
dataset = load_dataset("AdaptLLM/finance-tasks", name="Headline", split="test")

# 执行转换
output_data = generate_structured_data(dataset)
print(f"有效问答对数: {len(output_data)}")

# 保存结果
with open("headline_qa.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

# 验证输出
if os.path.exists("headline_qa.json"):
    print("✅ 文件已保存成功")
else:
    print("❌ 文件保存失败")




import json
from collections import defaultdict

# 读取数据
with open('headline_qa.json', 'r') as f:
    data = json.load(f)

# 分组
grouped = defaultdict(list)
for item in data:
    group_key = '-'.join(item['id'].split('-')[:2])  # 例如 headline-27
    grouped[group_key].append(item)

# 保留每组中含有 "Does" 的所有问句
cleaned = []
for items in grouped.values():
    # 找出含有 Does 的项
    with_does = [item for item in items if "Does" in item["Question"]]
    # 如果有，则只保留这些
    if with_does:
        cleaned.extend(with_does)
    else:
        # 如果全都不包含 Does，就保留所有（不删除任何）
        cleaned.extend(items)

# 修正id使其连续
for idx, item in enumerate(cleaned, 1):
    item['id'] = f"headline-{item['id'].split('-')[1]}-{idx}"

# 写入清洗结果
with open('cleaned_data.json', 'w') as f:
    json.dump(cleaned, f, indent=2)

print(f"总共保留了 {len(cleaned)} 条数据")

file_path = 'headline_qa.json'

# 删除文件
if os.path.exists(file_path):
    os.remove(file_path)
    print(f"文件 {file_path} 已被删除。")
else:
    print(f"文件 {file_path} 不存在。")