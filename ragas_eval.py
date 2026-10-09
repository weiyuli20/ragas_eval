from ragas import evaluate
from ragas.metrics import AnswerRelevancy, Faithfulness, ContextRecall, ContextPrecision

from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.dataset_schema import EvaluationDataset
import ast
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model=os.environ["model_name"],
    base_url=os.environ["base_url"],
    api_key=os.environ["DASHSCOPE_API_KEY"],)
evaluator_llm = LangchainLLMWrapper(llm)
evaluator_embeddings = LangchainEmbeddingsWrapper(
    DashScopeEmbeddings(
        model=os.getenv("embedding_model_name", "text-embedding-v4"),
        dashscope_api_key=os.environ["DASHSCOPE_API_KEY"],
    )
)

rag_results = pd.read_csv("ragas_evaluation_dataset.csv")
required_columns = {"user_input", "retrieved_contexts", "response", "reference"}
missing_columns = required_columns.difference(rag_results.columns)
if missing_columns:
    raise ValueError(f"Evaluation dataset is missing columns: {sorted(missing_columns)}")

def parse_contexts(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    contexts = ast.literal_eval(value)
    if not isinstance(contexts, list) or not all(
        isinstance(context, str) for context in contexts
    ):
        raise ValueError("retrieved_contexts must contain a list of strings")
    return contexts

rag_results["retrieved_contexts"] = rag_results["retrieved_contexts"].apply(
    parse_contexts
)
evaluation_dataset = EvaluationDataset.from_pandas(rag_results)

results = evaluate(
    dataset=evaluation_dataset,
    metrics=[
        AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            strictness=1,
        ),
        Faithfulness(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
        ContextPrecision(llm=evaluator_llm),
    ],
)

scores_path = "ragas_scores.csv"
scores_df = results.to_pandas()
scores_df.to_csv(scores_path, index=False, encoding="utf-8-sig")
print(f"Evaluation scores saved to {scores_path}")

metric_columns = [
    "answer_relevancy",
    "faithfulness",
    "context_recall",
    "context_precision",
]
mean_scores = scores_df[metric_columns].mean()
print("Average metric scores:")
for metric_name, mean_score in mean_scores.items():
    print(f"  {metric_name}: {mean_score:.4f}")
