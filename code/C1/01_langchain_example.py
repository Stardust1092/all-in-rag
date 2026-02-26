import os
# hugging face镜像设置，如果国内环境无法使用启用该设置
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

#导入必要的库、加载环境变量以及下载嵌入模型
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"

# 加载原始文档: 先定义Markdown文件的路径，然后使用TextLoader加载该文件作为知识源。
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

# 文本分块
# chunk_size: 每个块的最大字符数; chunk_overlap: 相邻块之间的重叠字符数
# 尝试修改这两个参数观察切分结果变化:
# 小块: chunk_size=100, chunk_overlap=20  -> 块数更多，每块内容更少
# 大块: chunk_size=1000, chunk_overlap=200 -> 块数更少，每块内容更多（默认值附近）
chunk_size = 1000
chunk_overlap = 200

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
chunks = text_splitter.split_documents(docs)

# 输出切分统计信息
print(f"\n===== 切分参数: chunk_size={chunk_size}, chunk_overlap={chunk_overlap} =====")
print(f"文档总块数: {len(chunks)}")
print(f"前3块的字符长度: {[len(c.page_content) for c in chunks[:3]]}")
print(f"\n--- 第1块内容 ---\n{chunks[0].page_content}")
print(f"\n--- 第2块内容 ---\n{chunks[1].page_content}")
print("=" * 60 + "\n")

# 中文嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}#启用嵌入归一化
)
  
# 构建向量存储
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:"""
                                          )

# 配置大语言模型
#配置大语言模型: 初始化 ChatOpenAI 客户端，配置所用模型（glm-4.7-flash-free）、生成答案的温度参数（temperature=0.7）、最大Token数 (max_tokens=2048) 以及API密钥（从环境变量加载）和 url。
# # 使用 AIHubmix
# llm = ChatOpenAI(
#     model="glm-4.7-flash-free",
#     temperature=0.7,
#     max_tokens=4096,
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://aihubmix.com/v1"
# )

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=4096,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 用户查询
question = "文中举了哪些例子？"

# 在向量存储中查询相关文档
#使用向量存储的similarity_search方法，根据用户问题在索引中查找最相关的 k (此处示例中 k=3) 个文本块。
retrieved_docs = vectorstore.similarity_search(question, k=3)

#准备上下文: 将检索到的多个文本块的页面内容 (doc.page_content) 合并成一个单一的字符串，并使用双换行符 ("\n\n") 分隔各个块，形成最终的上下文信息 (docs_content) 供大语言模型参考。
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

answer = llm.invoke(prompt.format(question=question, context=docs_content))
#print(answer)

#参数过滤掉得到content里的具体回答
print(answer.content)
