from typing import Dict, Any, Union, List, Optional, Tuple
from openai import OpenAI
import toml

def load_config(path: str = r"./config.toml", service: str = "openai") -> Dict[str, Any]:
    """加载模型服务配置
    
    Args:
        path: 配置文件路径
        service: 服务类型，'openai' 或 'deepseek'
        
    Returns:
        包含模型配置的字典
    """
    cfg = toml.load(path)
    service_cfg = cfg[service]
    
    return {
        "api_key": service_cfg["api_key"],
        "base_url": service_cfg.get("base_url"),
        "model": service_cfg.get("model"),
    }

def create_openai_client(api_key: str, base_url: str | None) -> OpenAI:
    """创建OpenAI客户端
    
    Args:
        api_key: API密钥
        base_url: API基础URL
        
    Returns:
        OpenAI客户端实例
    """
    return OpenAI(api_key=api_key, base_url=base_url)


def create_deepseek_client(api_key: str, base_url: str | None) -> OpenAI:
    """创建DeepSeek客户端
    
    Args:
        api_key: API密钥
        base_url: API基础URL
        
    Returns:
        OpenAI客户端实例（兼容DeepSeek API）
    """
    return OpenAI(api_key=api_key, base_url=base_url)

def chat_openai(
    content: str = "",
    system: str = "你是乐于助人的AI,请使用中文回答问题",
    return_type: str = "text",
    stream: bool = False,
    **kwargs
) -> Union[Dict[str, Any], str, List[Any], None]:
    """
    调用OpenAI API
    
    Args:
        content: 用户输入的内容
        system: 系统提示词
        return_type: 返回类型，'json'返回原始响应，'text'返回纯文本
        stream: 是否使用流式调用，默认为False
        **kwargs: 其他传递给API的参数
        
    Returns:
        - 非流式调用: 返回完整响应或文本
        - 流式调用: 返回响应块列表或逐步输出文本
    """
    cfg = load_config()
    client = create_openai_client(cfg["api_key"], cfg.get("base_url"))
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]
    
    # 准备API参数
    api_params = {
        "model": cfg["model"],
        "messages": msgs,
        "stream": stream,
        **kwargs
    }
    
    if stream:
        # 流式调用
        response = client.chat.completions.create(**api_params)
        
        if return_type == "json":
            chunk_list = []
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    chunk_list.append(chunk)
                    print(chunk)
            return chunk_list
        elif return_type == "text":
            full_content = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_content += content
                    print(content, end="", flush=True)
            print()  # 添加换行
            return full_content
        else:
            raise ValueError("return_type 必须是 'json' 或 'text'")
    else:
        # 非流式调用
        response = client.chat.completions.create(**api_params)
        
        if return_type == "json":
            return response
        elif return_type == "text":
            return response.choices[0].message.content
        else:
            raise ValueError("return_type 必须是 'json' 或 'text'")

def chat_deepseek(
    content: str = "",
    system: str = "你是DeepSeek AI助手，请使用中文回答问题",
    return_type: str = "text",
    stream: bool = False,
    model: str = None,
    messages: List[Dict[str, str]] = None,
    **kwargs
) -> Union[Dict[str, Any], str, Tuple[str, str], List[Any], None]:
    """调用DeepSeek API，支持推理模型
    
    Args:
        content: 用户输入的内容
        system: 系统提示词
        return_type: 返回类型，'json'返回原始响应，'text'返回纯文本，'both'返回(content, reasoning_content)
        stream: 是否使用流式调用，默认为False
        model: 指定模型，如果为None则使用配置文件中的模型
        messages: 消息历史，如果提供则忽略content和system参数
        **kwargs: 其他传递给API的参数
        
    Returns:
        - 非流式调用: 
            - return_type='json': 返回完整响应
            - return_type='text': 返回纯文本内容
            - return_type='both': 返回元组 (content, reasoning_content)
        - 流式调用: 
            - return_type='json': 返回响应块列表
            - return_type='text': 打印并返回完整内容
    """
    # 加载DeepSeek配置
    cfg = load_config(service="deepseek")
    client = create_deepseek_client(cfg["api_key"], cfg.get("base_url"))
    
    # 构建消息
    if messages is None:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
    
    # 准备API参数
    api_params = {
        "model": model or cfg["model"],
        "messages": messages,
        "stream": stream,
        **kwargs
    }
    
    if stream:
        # 流式调用
        response = client.chat.completions.create(**api_params)
        
        if return_type == "json":
            chunk_list = []
            for chunk in response:
                if chunk.choices[0].delta is not None:
                    chunk_list.append(chunk)
                    print(chunk)
            return chunk_list
        else:
            full_content = ""
            reasoning_content = ""
            
            for chunk in response:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    reasoning_content += delta.reasoning_content
                    print(f"[推理] {delta.reasoning_content}", end="", flush=True)
                elif delta.content:
                    full_content += delta.content
                    print(delta.content, end="", flush=True)
            
            print()  # 添加换行
            
            if return_type == "both":
                return full_content, reasoning_content
            return full_content
    else:
        # 非流式调用
        response = client.chat.completions.create(**api_params)
        
        if return_type == "json":
            return response
        elif return_type == "both":
            content = response.choices[0].message.content
            reasoning_content = getattr(response.choices[0].message, 'reasoning_content', '')
            return content, reasoning_content
        else:
            return response.choices[0].message.content


if __name__ == "__main__":
    # 测试OpenAI API
    print("=== 测试 OpenAI API ===")
    # 默认非流式调用（返回文本）
    print("\n=== 默认非流式调用 ===")
    result = chat_openai("4 4 6 8如何计算得出24?")
    print(result)
    
    # 非流式调用（返回JSON）
    print("\n=== 非流式调用 (JSON) ===")
    result_json = chat_openai("4 4 6 8如何计算得出24?", return_type="json")
    print(result_json)
    
    # 流式调用（返回文本）
    print("\n=== 流式调用 (Text) ===")
    chat_openai("4 4 6 8如何计算得出24?", stream=True)
    
    # 测试DeepSeek API
    print("\n\n=== 测试 DeepSeek API ===")
    # 默认非流式调用（返回文本）
    print("\n=== 默认非流式调用 ===")
    result = chat_deepseek("4 4 6 8如何计算得出24?")
    print(result)
    
    # 非流式调用（返回JSON）
    print("\n=== 非流式调用 (JSON) ===")
    result_json = chat_deepseek("4 4 6 8如何计算得出24?", return_type="json")
    print(result_json)
    
    # 流式调用（返回文本）
    print("\n=== 流式调用 (Text) ===")
    chat_deepseek("4 4 6 8如何计算得出24?", stream=True)
