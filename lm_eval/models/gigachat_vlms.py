import base64
import copy
import json
import logging
import warnings
from io import BytesIO
from typing import Dict, List, Optional, Union

import requests
from PIL import Image
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.api_models import JsonChatStr
from lm_eval.models.openai_completions import LocalChatCompletion
from lm_eval.models.utils import Collator


DEFAULT_IMAGE_PLACEHOLDER = "<image>"
GIGAVISION_IMAGE_PLACEHOLDER = "[image_token]"


warnings.filterwarnings(
    "ignore"
)  # turn off insecure connection warning if verify_certificate=False


eval_logger = logging.getLogger(__name__)


def pil_image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt)
    byte_data = buf.getvalue()
    buf.close()
    return convert_to_base64(byte_data)


def convert_to_base64(img_bytes):
    return base64.b64encode(img_bytes).decode("utf-8")


@register_model("gigachat-vlms")
class GigaChatVLMLocal(LocalChatCompletion):
    MULTIMODAL = True

    def __init__(
        self,
        base_url=None,
        verify_certificate=False,
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
            verify_certificate=verify_certificate,
            **kwargs,
        )

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        generate=True,
        gen_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        if generate:
            return {
                "messages": messages,
                "model": self.model,
                **gen_kwargs,
            }
        else:
            return None

    def create_message(
        self,
        messages: Union[List[List[int]], List[str], List[JsonChatStr]],
        multimodal_args: list,
        generate=False,
    ) -> Union[List[List[int]], List[dict], List[str], str]:
        """Helper method to transform the prompt into the expected API input format. messages consist of batched requests"""
        if isinstance(messages[0], JsonChatStr):
            # for chat completions we need to decode the json string to list[dict,...]
            assert self._batch_size == 1, (
                "non-tokenized chat requests are only supported with batch_size=1"
            )
            # list[dict["role":..., "content":...],...]
            text = json.loads(messages[0].prompt)

            msg = []

            if len(text) == 2:
                # if has system role
                msg.extend([text[0]])
            # user role add
            msg.extend(
                [self.create_multimodal_messages(text[-1]["content"], multimodal_args)]
            )

            return msg

        return messages

    def create_multimodal_messages(self, message, multimodal_args):
        visuals = multimodal_args[0]["visual"]
        # if isinstance(visuals, list):
        #     visuals = visuals[0]

        visuals = [pil_image_to_bytes(elem) for elem in visuals]

        return {
            "role": "user",
            "content": message.replace(
                DEFAULT_IMAGE_PLACEHOLDER, GIGAVISION_IMAGE_PLACEHOLDER
            ),
            "attachments": visuals,
        }

    def model_call(
        self,
        messages: Union[List[List[int]], List[str], List[JsonChatStr]],
        multimodal_args: list,
        *,
        generate: bool = True,
        gen_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Optional[dict]:
        # !!! Copy: shared dict for each request, need new object !!!
        gen_kwargs = copy.deepcopy(gen_kwargs)
        try:
            js = self._create_payload(
                self.create_message(messages, multimodal_args),
                gen_kwargs=gen_kwargs,
                **kwargs,
            )

            response = requests.post(self.base_url, json=js)
            if not response.ok:
                eval_logger.warning(
                    f"API request failed with error message: {response.text}. Retrying..."
                )
            response.raise_for_status()
            return response.json()
        except RetryError:
            eval_logger.error(
                "API request failed after multiple retries. Please check the API status."
            )
            return None

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        res = []

        def _collate_gen(_requests):
            # sort by the length of the non-tokenized contexts
            return -len(_requests[0])

        # Let the API deal with tokenization
        requests, all_gen_kwargs, multimodal_args = zip(*(req.args for req in requests))
        if self.tokenized_requests:
            encodings_list = self.tok_encode(
                requests, add_special_tokens=self.add_bos_token
            )
        else:
            encodings_list = [None] * len(requests)
        requests = [
            (a, b, c, d)
            for a, b, c, d in zip(
                requests, all_gen_kwargs, encodings_list, multimodal_args
            )
        ]

        re_ord = Collator(
            requests,
            sort_fn=_collate_gen,
            group_by="gen_kwargs",
        )
        chunked = re_ord.get_batched(n=self._batch_size, batch_fn=None)

        pbar = tqdm(desc="Requesting GigaVision Local API", total=len(requests))
        for chunk in chunked:
            contexts, all_gen_kwargs, encodings_list, mm_args = zip(*chunk)

            eval_logger.info(
                "Tokenized requests are disabled. Context + generation length is not checked."
            )

            gen_kwargs = {"stream": False}
            temperature = all_gen_kwargs[0].get("temperature", None)
            max_tokens = all_gen_kwargs[0].get("max_gen_toks", self._max_gen_toks)
            top_p = all_gen_kwargs[0].get("top_p", None)

            if temperature:
                gen_kwargs["temperature"] = temperature
            if max_tokens:
                gen_kwargs["max_tokens"] = max_tokens
            if top_p:
                gen_kwargs["top_p"] = top_p

            req = contexts
            outputs = retry(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=0.5, min=1, max=10),
                reraise=True,
            )(self.model_call)(
                messages=req,
                multimodal_args=mm_args,
                generate=True,
                gen_kwargs=copy.deepcopy(gen_kwargs),
            )
            for generated_text, context in zip(
                self.parse_generations(
                    outputs=outputs,
                    contexts=contexts,
                ),
                contexts,
            ):
                if generated_text is not None:
                    res.append(generated_text)

                    # partial caching
                    if context is not None:
                        self.cache_hook.add_partial(
                            "generate_until",
                            (context, all_gen_kwargs[0]),
                            generated_text,
                        )
                        pbar.update(1)

        return re_ord.get_original(res)
