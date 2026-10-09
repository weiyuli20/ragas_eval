from typing import List
from tqdm import tqdm
from openai import OpenAI
from pymilvus import MilvusClient
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()


class RAG:
    """
    RAG (Retrieval-Augmented Generation) class built upon OpenAI and Milvus.
    """

    def __init__(self, openai_client: OpenAI, milvus_client: MilvusClient):
        self._prepare_openai(openai_client)
        self._prepare_milvus(milvus_client)

    def _emb_text(self, text: str) -> List[float]:
        return self.embedding_client.embed_query(text)

    def _prepare_openai(
        self,
        openai_client: OpenAI,
        embedding_model: str = None,
        llm_model: str = "qwen3-8b",
    ):
        self.openai_client = openai_client
        self.embedding_model = embedding_model or os.getenv(
            "embedding_model_name", "text-embedding-v4"
        )
        self.embedding_client = DashScopeEmbeddings(
            model=self.embedding_model,
            dashscope_api_key=os.environ["DASHSCOPE_API_KEY"],
        )
        self.llm_model = llm_model
        self.SYSTEM_PROMPT = """
Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
"""
        self.USER_PROMPT = """
Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
<context>
{context}
</context>
<question>
{question}
</question>
"""

    def _prepare_milvus(
        self, milvus_client: MilvusClient, collection_name: str = "rag_collection"
    ):
        self.milvus_client = milvus_client
        self.collection_name = collection_name
        if self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.drop_collection(self.collection_name)
        embedding_dim = len(self._emb_text("foo"))
        self.milvus_client.create_collection(
            collection_name=self.collection_name,
            dimension=embedding_dim,
            metric_type="IP",  # Inner product distance
            consistency_level="Bounded",  # Strong consistency level
        )

    def load(self, texts: List[str]):
        """
        Load the text data into Milvus.
        """
        data = []
        for i, line in enumerate(tqdm(texts, desc="Creating embeddings")):
            data.append({"id": i, "vector": self._emb_text(line), "text": line})

        self.milvus_client.insert(collection_name=self.collection_name, data=data)

    def retrieve(self, question: str, top_k: int = 3) -> List[str]:
        """
        Retrieve the most similar text data to the given question.
        """
        search_res = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[self._emb_text(question)],
            limit=top_k,
            search_params={"metric_type": "IP", "params": {}},  # Inner product distance
            output_fields=["text"],  # Return the text field
        )
        retrieved_texts = [res["entity"]["text"] for res in search_res[0]]
        return retrieved_texts[:top_k]

    def answer(
        self,
        question: str,
        retrieval_top_k: int = 3,
        return_retrieved_text: bool = False,
    ):
        """
        Answer the given question with the retrieved knowledge.
        """
        retrieved_texts = self.retrieve(question, top_k=retrieval_top_k)
        user_prompt = self.USER_PROMPT.format(
            context="\n".join(retrieved_texts), question=question
        )
        response = self.openai_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        if not return_retrieved_text:
            return response.choices[0].message.content
        else:
            return response.choices[0].message.content, retrieved_texts

openai_client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url = os.environ["base_url"]

)
milvus_client = MilvusClient(uri="./milvus_demo.db")

my_rag = RAG(openai_client=openai_client, milvus_client=milvus_client)

path = "Sample_Docs_Markdown/"
loader = DirectoryLoader(path, glob="**/*.md")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
text_chunks = text_splitter.split_documents(docs)
text_lines = [chunk.page_content for chunk in text_chunks]
my_rag.load(text_lines)

question = "whats a gItLab ally lab?"
answer = my_rag.answer(question, return_retrieved_text=True)
print("Q:",question)
print("A:",answer)

#构建数据集
import pandas as pd
from ragas.dataset_schema import EvaluationDataset

df = pd.read_csv("ragas_testset.csv")
required_columns = {"user_input", "reference"}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
    raise ValueError(f"Test set is missing columns: {sorted(missing_columns)}")

user_input_list = []
retrieved_contexts_list = []
response_list = []
reference_list = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Running RAG on test set"):
    question = str(row["user_input"])
    response, retrieved_contexts = my_rag.answer(
        question,
        return_retrieved_text=True,
    )
    user_input_list.append(question)
    retrieved_contexts_list.append(retrieved_contexts)
    response_list.append(response or "")
    reference_list.append(str(row["reference"]))

evaluation_df = pd.DataFrame(
    {
        "user_input": user_input_list,
        "retrieved_contexts": retrieved_contexts_list,
        "response": response_list,
        "reference": reference_list,
    }
)
rag_results = EvaluationDataset.from_pandas(evaluation_df)
print(f"Built evaluation dataset with {len(evaluation_df)} samples")
output_path = "ragas_evaluation_dataset.csv"
rag_results.to_pandas().to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"Final evaluation dataset saved to {output_path}")





