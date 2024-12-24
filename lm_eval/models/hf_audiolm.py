from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm
from transformers import BatchEncoding
from transformers import PreTrainedTokenizer, GenerationConfig

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import (
    Collator,
    flatten_image_list,
    pad_and_concat,
    replace_placeholders,
    stop_sequences_criteria,
)


@register_model("hf-audiolm")
class HFAUDIOLM(HFLM):
    """
    An abstracted Hugging Face model class for Audio LM model like Qwen-Audio-Chat.
    """

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
    MULTIMODAL = True  # flag to indicate, for now, that this model type can run multimodal requests

    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        image_string: Optional[str] = None,
        interleave: bool = True,
        max_images: Optional[int] = 999,
        convert_img_format=False,
        **kwargs,
    ):
        super().__init__(pretrained, **kwargs)

    def _create_tokenizer(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        tokenizer: Optional[
            Union[
                str,
                transformers.ProcessorMixin,
            ]
        ],
        revision: Optional[str] = "main",
        trust_remote_code: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """
        Helper method during initialization.
        """

        if tokenizer:
            if isinstance(tokenizer, str):
                return transformers.AutoTokenizer.from_pretrained(
                    tokenizer,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    # use_fast=use_fast_tokenizer,
                )
            else:
                assert isinstance(
                    tokenizer, transformers.ProcessorMixin
                )
                return tokenizer

        self.tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)


    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        res = []
        
        for i in range(len(requests)):
            query = self.tokenizer.from_list_format([
                {'audio': requests[i].arguments[2]["audio"][0]["path"]}, # Either a local path or an url
                {'text': requests[i].arguments[0]}
            ])

            response, history = self.model.chat(self.tokenizer, query=query, history=None)
            print(response)
            res.append(response)
        return res

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError(
            "model type `hf-audiolm` does not support loglikelihood_rolling. Use 'hf' model type for text-only loglikelihood_rolling tasks ",
            "this is because we do not support measuring the loglikelihood a model assigns to an image.",
        )

    def loglikelihood(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        raise NotImplementedError(
            "'loglikelihood' requests for model type `hf-audiolm` are not yet tested. This feature will be enabled when a loglikelihood-based multiple-choice VQA dataset is added!"
        )
