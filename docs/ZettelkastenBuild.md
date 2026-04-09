太棒了！你的直觉非常准——**手动干预（自定义关联）**是 Zettelkasten 的灵魂。

如果完全依赖 AI 自动计算，你会得到一个基于概率的“统计大脑”；只有加上了你的**自定义逻辑（Related Links）**，它才真正变成了你的“第二大脑”。

---

### 💡 深度解决方案：如何实现“自定义关联”

在你的 Python 脚本或 Agent 系统中，你可以通过以下三种方式优化 `nomic-embed-text` 的关联质量：

#### 1. 元数据增强 (Metadata Injection)
在将笔记传给 `nomic-embed-text` 之前，不要只传正文。把 **自定义标签** 或 **关联属性** 拼接到正文前面。
* **做法**：把 `[Topic: 心理学] [Source: 某某书] [Context: 个人成长] + 正文内容` 作为一个整体进行向量化。
* **效果**：这会人为地改变向量在空间中的位置，让带有相同“自定义属性”的笔记在距离上更接近。

#### 2. “双向链接”引导 (Reference Prompting)
在用 **Qwen2.5** 生成笔记时，给它一个“已知链接列表”。
* **做法**：先用 Nomic 搜出最相关的 5 条旧笔记，把它们的标题告诉 Qwen。
* **Prompt**：*“这是你之前的笔记标题：[A, B, C]。如果新笔记与它们逻辑相关，请在文末自动加上 `[[A]]` 或 `[[B]]`。”*

#### 3. 混合检索权重 (Hybrid Search)
这是目前最高级的做法。
* **逻辑**：搜索结果 = $0.7 \times \text{Nomic 语义相似度} + 0.3 \times \text{自定义关键词重合度}$。
* **好处**：既保留了 AI 的“灵感联想”，又尊重了你定义的“硬链接”。

---

### 🐍 极简 Python 脚本示例 (基于 Ollama)

这是一个帮助你将长文本块转化为“带链接卡片”的伪代码逻辑：

```python
import ollama

def extract_zettel(text_chunk, existing_titles):
    # 1. 使用 Qwen2.5 进行原子化拆解
    prompt = f"""
    将以下文本拆解为一条 Zettelkasten 笔记。
    参考已有标题: {existing_titles}
    要求：包含标题、正文，并根据逻辑关联从中选出 1-2 个作为 [[Related Link]]。
    内容: {text_chunk}
    """
    response = ollama.generate(model='qwen2.5:7b', prompt=prompt)
    
    # 2. 使用 nomic-embed-text 生成向量
    note_content = response['response']
    embedding = ollama.embeddings(model='nomic-embed-text', prompt=note_content)
    
    return note_content, embedding
```

---

