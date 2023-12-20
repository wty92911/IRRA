import spacy
from utils.simple_tokenizer import SimpleTokenizer
# 加载英语模型
nlp = spacy.load("en_core_web_sm")
import torch
def extract_and_mask_phrases(text):
    # 处理文本
    doc = nlp(text)

    # 提取名词短语
    phrases = [chunk.text for chunk in doc.noun_chunks]

    # 遮蔽短语
    masked_text = text
    for phrase in phrases:
        masked_text = masked_text.replace(phrase, "[MASK]")

    return masked_text, phrases

# 示例文本
example_text = "The woman is wearing a black top and grey pants. She is carrying some papers or a book and has a purse over her shoulder. She is has black shoulder length hair and no bangs."
tokenizer = SimpleTokenizer()
# 提取并遮蔽短语
tokens = torch.tensor(tokenizer.encode(example_text))
print(tokens)
masked_text, extracted_phrases = extract_and_mask_phrases(example_text)
    #在tokens中找到连续的一部分是phrase_tokens mask
for phrase in extracted_phrases:
    phrase_tokens = torch.tensor(tokenizer.encode(phrase))
    for i, token in enumerate(tokens):
        if torch.equal(tokens[i: i + len(phrase_tokens)], phrase_tokens):
            tokens[i: i + len(phrase_tokens)] = 0
print(tokens)
# 输出结果
# print("原始文本:", example_text)
# print("提取的短语:", extracted_phrases)

# print("遮蔽后的文本:", masked_text)
