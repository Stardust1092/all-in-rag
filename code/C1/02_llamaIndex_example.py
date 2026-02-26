import os
# 如需使用 huggingface 镜像（国内环境），可取消注释下面一行
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'

# 加载环境变量（如 API KEY）
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# # 使用 AIHubmix（可选）
# Settings.llm = OpenAILike(
#     model="glm-4.7-flash-free",
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     api_base="https://aihubmix.com/v1",
#     is_chat_model=True
# )

# 配置 LLM（大语言模型）为 deepseek-chat，API KEY 从环境变量读取
Settings.llm = OpenAILike(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base="https://api.deepseek.com/v1",
    is_chat_model=True
)

# 配置嵌入模型（用于文本向量化）
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 加载文档（单文件）
docs = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

# 构建向量索引（将文档转为向量，便于检索）
index = VectorStoreIndex.from_documents(docs)

# 创建查询引擎（用于问答）
query_engine = index.as_query_engine()

# 输出当前使用的提示模板
print(query_engine.get_prompts())

# 执行查询，输出答案
print(query_engine.query("文中举了哪些例子?"))