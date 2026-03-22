from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

from prompt_temp import ChemistryPrompt

import pandas as pd
import random
import re
from typing import Union, Literal


@OPERATOR_REGISTRY.register()
class ChemistryQuestionGenerator(OperatorABC):
    def __init__(self,
                num_prompts: int = 1,
                llm_serving: LLMServingABC = None,
                prompt_template = ChemistryPrompt(),
                save_mode: Union[str, Literal["full", "synth"]] = "full"  #select "full" or "synth"
                ):
        self.logger = get_logger()
        self.prompts = prompt_template
        self.num_prompts = num_prompts
        self.llm_serving = llm_serving
        self.save_mode = save_mode

        if self.num_prompts not in range(1,6):
            self.logger.debug("num_prompts must be an integer between 1 and 5 (inclusive)")
        
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key]
        forbidden_keys = ["Synth_or_Input"]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten: {conflict}")

        
    def _reformat_prompt(self, dataframe):
        """
        Reformat the prompts in the dataframe to generate questions based on num_prompts.
        """
        diversity_mode = [
            "1, 2, 3",       # Practical Lab Focus
            "1, 4, 5",       # Theoretical/Mechanistic Focus
            "2, 4, 5",       # Industrial/Complexity Focus
            "1, 3, 5",       # Analytical/Deductive Focus
            "1, 2, 3, 4, 5"  # Extreme Complexity
        ]
        formatted_prompts = []
        selected_items = random.sample(diversity_mode, self.num_prompts)
        for selected_item in selected_items:
            for question in dataframe[self.input_key]:
                used_prompt = self.prompts.chemisry_question_generate(selected_item, question)
                formatted_prompts.append(used_prompt.strip())
        return formatted_prompts

    def _parse_response(self, response: str):
        # useful for reasoning llms. If response is in format of Deepseek thinking: <think>...</think><answer>...</answer>, keep only the answer part.
        pattern = r"<think>(.*?)</think><answer>(.*?)</answer>"
        matches = re.findall(pattern, response)
        if matches:
            return matches[0][1]
        else:
            return response
    
    def run(
            self, 
            storage: DataFlowStorage, 
            input_key: str,
            output_synth_or_input_flag: str = "Synth_or_Input"
            ):
        """
        Run the question generation process.
        """
        self.input_key = input_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)
        formatted_prompts = self._reformat_prompt(dataframe)
        responses = self.llm_serving.generate_from_input(formatted_prompts)
        responses = [self._parse_response(response) for response in responses]

        new_rows = pd.DataFrame({
            input_key: responses,
        })
        new_rows[output_synth_or_input_flag] = "synth"
        dataframe[output_synth_or_input_flag] = "input"

        if self.save_mode == "full":
            dataframe = pd.concat([dataframe, new_rows], ignore_index=True)
        elif self.save_mode == "synth":
            dataframe = new_rows
        dataframe = dataframe[dataframe[input_key].notna()]
        dataframe = dataframe[dataframe[input_key] != ""]

        output_file = storage.write(dataframe)
        self.logger.info(f"Generated questions saved to {output_file}")

        return [input_key, output_synth_or_input_flag]
