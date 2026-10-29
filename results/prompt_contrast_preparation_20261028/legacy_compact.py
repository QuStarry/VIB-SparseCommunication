def render(prompt, history):
    history_text = '\n'.join(history) if history else '(无历史讨论)'
    prompt_text = f'请阅读团队讨论历史，补全并实现以下 Python 函数。\n\n【需求与签名】:\n{prompt}\n\n【团队讨论历史】:\n{history_text}\n\n【极其重要的输出要求】:\n1. 仅输出最终可执行的 Python 代码实现。\n2. 必须包含完整的函数定义并直接可被调用。\n3. 不要包含任何 markdown 说明文字或解说，直接写代码。'
    return prompt_text
