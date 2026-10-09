import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import DashScopeEmbeddings

load_dotenv()

path = "Sample_Docs_Markdown/"
loader = DirectoryLoader(path, glob="**/*.md")
docs = loader.load()

from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.graph import NodeType
from ragas.testset.transforms import HeadlineSplitter, default_transforms
from ragas.utils import num_tokens_from_string


generator_llm = LangchainLLMWrapper(ChatOpenAI(
    model=os.environ["model_name"],
    base_url=os.environ["base_url"],
    api_key=os.environ["DASHSCOPE_API_KEY"],
    ))

generator_embeddings = LangchainEmbeddingsWrapper(
    DashScopeEmbeddings(
        model=os.getenv("embedding_model_name", "text-embedding-v4"),
        dashscope_api_key=os.environ["DASHSCOPE_API_KEY"],
    )
)


#生成测试集
from ragas.testset import TestsetGenerator

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
transforms = default_transforms(docs, generator_llm, generator_embeddings)
for transform in transforms:
    if isinstance(transform, HeadlineSplitter):
        transform.filter_nodes = lambda node: (
            node.type == NodeType.DOCUMENT
            and num_tokens_from_string(node.properties["page_content"]) > 500
        )

dataset = generator.generate_with_langchain_docs(
    docs, testset_size=10, transforms=transforms
)

output_path = "ragas_testset.csv"
dataset.to_pandas().to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"Test set saved to {output_path}")