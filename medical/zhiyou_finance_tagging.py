from dataflow.pipeline import BatchedPipelineABC
from dataflow.operators.reasoning.generate.reasoning_trajectory_generator import ReasoningTrajectoryGenerator
from dataflow.utils.storage import BatchedFileStorage
from dataflow.serving import APILLMServing_request
from dataflow.prompts.reasoning.diy import DiyQuestionSynthesisPrompt


BATCH_SIZE = 500

# DIY prompt for finance answer generation
DIY_PROMPT_ANSWER = """
    You are a finance-domain task tagger. From a strictly financial perspective, carefully analyze and think through the input (question, instruction, or data description) to identify ALL finance-related tasks it involves. Do not answer the question itself—only tag the tasks.

    # Requirements
    Select task names from the unified list below whenever possible. If a task is not covered, define the task based on your financial knowledge.

    Unified Finance Task List (Benchmark + Common Finance Tasks):
    - MonetaryPolicyStanceClassification
    - FinancialNewsSentimentAnalysis
    - StockPriceMovementPrediction
    - FinancialTextClassification
    - InformationExtraction
    - EventImpactAnalysis
    - MarketTrendPrediction
    - AssetPriceForecasting
    - VolatilityEstimation
    - RiskAssessment
    - CreditEvaluation
    - FraudAnomalyDetection
    - MacroEconomicAnalysis
    - PortfolioOptimization
    - AssetAllocationRecommendation
    - TradingSignalGeneration
    - FinancialQuestionAnswering
    - EarningsCallAnalysis
    - AnalystRatingRecommendationAnalysis
    - CompanyFundamentalsAnalysis
    - ValuationEstimation
    - LiquidityAnalysis
    - CorrelationDependencyAnalysis
    ...

    Tagging Rules:
    1) Think step-by-step internally from a finance perspective, but DO NOT output your reasoning.
    2) Output ALL tasks that are materially required by the input. Include both primary and secondary tasks if they are explicitly involved.
    3) If the input includes a substep (e.g., extract events/entities, then predict movement), include both tasks.
    4) Prefer existing task names from the list; only create a new one when necessary.
    5) Avoid redundant or near-duplicate task names.

    Output Format (STRICT):
    - Output only a single JSON-style list of strings, e.g. ["TaskA","TaskB"]
    - No extra text, no punctuation outside the list.

    # Input
    The financial related input is: {question}
    """


class ReasoningTaggingText(BatchedPipelineABC):
    def __init__(
        self,
        api_url: str,
        model_name: str,
        first_entry_file_name: str,
        cache_path: str,
        file_name_prefix: str,
        cache_type: str,
        input_key: str,
        output_key: str = "generated_text",
        max_workers: int = 10,
    ):
        """
        Reasoning answer text pipeline based on ReasoningAnswerGenerator.

        Args:
            api_url: LLM 服务的 HTTP API 地址，例如 "http://localhost:30000/v1/chat/completions"
            model_name: 模型名称，例如 "qwen32b"
            first_entry_file_name: 首个输入文件路径（json/jsonl/csv/parquet/pickle）
            cache_path: 中间结果缓存目录
            file_name_prefix: 中间结果文件前缀
            cache_type: 中间结果文件格式，例如 "jsonl"
            input_key: dataframe 中作为指令/问题的列名，例如 "question" 或 "instruction"
            output_key: 生成回答写入的列名，默认为 "generated_text"
            max_workers: LLM 服务的最大并发数，默认为 10
        """
        super().__init__()
        
        # storage 负责读写数据文件
        self.storage = BatchedFileStorage(
            first_entry_file_name=first_entry_file_name,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )

        # LLM 服务
        self.llm_serving = APILLMServing_request(
            api_url=api_url,
            model_name=model_name,
            max_workers=max_workers,
            read_timeout=180.0,
            connect_timeout=60.0
        )

        # 使用 DIY prompt 进行回答生成
        self.prompt = DiyQuestionSynthesisPrompt(DIY_PROMPT_ANSWER)

        # 推理答案生成算子
        self.generator = ReasoningTrajectoryGenerator(
            llm_serving=self.llm_serving,
            prompt_template=self.prompt,
        )

        self.input_key = input_key
        self.output_key = output_key

    def forward(self):
        """
        Forward pass of the reasoning answer text pipeline.
        """
        if not self.generator:
            raise ValueError("Generator not initialized")

        # 这里只使用一个算子步骤：ReasoningAnswerGenerator
        self.generator.run(
            storage=self.storage.step(),
            input_key=self.input_key,
            output_key=self.output_key,
        )


if __name__ == "__main__":
    # 使用示例：请根据具体数据文件和字段名修改下面参数
    # 例如：MedQA-USMLE-4-options/phrases_no_exclude_train.jsonl 中
    # 如果问题字段名为 "question" 或 "instruction"，请对应修改 input_key
    pipeline = ReasoningTaggingText(
        api_url="https://kspmas.ksyun.com/v1/chat/completions",
        model_name=s,
        first_entry_file_name="/share/project/xzy_datas_models/infinity/Finance/Datasets/finance-alpaca/finance_alpaca.jsonl",
        cache_path="/share/project/xzy_datas_models/infinity/Finance/seeds",
        file_name_prefix="finance_alpaca_tagging",
        cache_type="jsonl",
        input_key="instruction",  # TODO: 根据实际数据字段名修改，例如 "question", "input", 等
        output_key="tags",
        max_workers=15,
    )
    pipeline.compile()
    pipeline.forward(batch_size=BATCH_SIZE, resume_from_last=True)

# api key = 8407460c-9a3d-4a32-bb0d-43e91a74304f
# export DF_API_KEY=8407460c-9a3d-4a32-bb0d-43e91a74304f
