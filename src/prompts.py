"""
Prompt templates for different query scenarios.
"""

# ── QA (default) ──

QA_TEMPLATE = """你是一个知识库问答助手。

请基于以下文档内容回答问题。如果文档中没有足够信息，请明确说"文档中未找到相关信息"，不要编造答案。

文档内容：
{context}

问题：{question}

要求：
- 回答尽可能准确，基于文档内容
- 如果引用了文档中的具体内容，请标注来源文件名（如 [来源: filename.pdf]）
- 不要添加文档中不存在的信息

回答："""

# ── Summary ──

SUMMARY_TEMPLATE = """请总结以下文档内容的核心要点。

文档内容：
{context}

要求：
- 用要点形式列出核心结论
- 每个要点一句话，简洁明了
- 保持原文的关键信息不丢失
- 如果存在多个主题，按主题分组

总结："""

# ── Code Explanation ──

CODE_TEMPLATE = """以下文档中包含代码相关内容。

文档内容：
{context}

问题：{question}

要求：
- 解释相关代码的逻辑和用法
- 如有必要，给出简单的使用示例
- 说明代码的输入输出和前置条件

回答："""


def detect_question_type(question: str) -> str:
    """Detect the type of question based on keywords.

    Returns one of: ``"qa"``, ``"summary"``, ``"code"``.
    """
    summary_keywords = ["总结", "概括", "归纳", "要点", "摘要", "summary", "summarize"]
    code_keywords = ["代码", "实现", "函数", "代码示例", "用法", "api", "接口", "code", "function"]

    q_lower = question.lower()
    for kw in summary_keywords:
        if kw in q_lower:
            return "summary"
    for kw in code_keywords:
        if kw in q_lower:
            return "code"
    return "qa"


def build_prompt(question: str, context: str, template_name: str | None = None) -> str:
    """Build a prompt by selecting the appropriate template.

    Args:
        question: The user's query.
        context: Retrieved chunk texts joined together.
        template_name: One of ``"qa"``, ``"summary"``, ``"code"``.
                       If ``None``, auto-detect from question content.

    Returns:
        The formatted prompt string.
    """
    if template_name is None:
        template_name = detect_question_type(question)

    templates = {
        "qa": QA_TEMPLATE,
        "summary": SUMMARY_TEMPLATE,
        "code": CODE_TEMPLATE,
    }

    template = templates.get(template_name, QA_TEMPLATE)
    return template.format(question=question, context=context)
