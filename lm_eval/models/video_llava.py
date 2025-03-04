import math
from datetime import timedelta
from typing import List, Optional, Tuple, Union
import copy

import numpy as np
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from loguru import logger
from tqdm import tqdm
import transformers
from transformers import AutoConfig, PreTrainedModel, AutoModelForPreTraining, AutoProcessor
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.models.huggingface import HFLM
from lm_eval.models.hf_vlms import HFMultimodalLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator, stop_sequences_criteria, handle_stop_sequences

eval_logger = utils.eval_logger



@register_model("video_llava")
class VideoLLaVA(HFMultimodalLM):
    MULTIMODAL = True
    AUTO_MODEL_CLASS = AutoModelForPreTraining
    
    def __init__(
        self,
        pretrained: Union[str, PreTrainedModel],
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = False,
        revision=None,
        attn_implementation=(
            "sdpa" if torch.__version__ > "2.1.2" else "eager"
        ),  # inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
        device_map="cuda:0",
        conv_template="llava_v1",
        use_cache=True,
        truncate_context=False,
        num_frames: int = 32,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        **kwargs,
    ) -> None:
        super().__init__(pretrained, **kwargs)
        # accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        # accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        # if accelerator.num_processes > 1:
        #     self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        #     self.device_map = f"cuda:{accelerator.local_process_index}"
        # elif accelerator.num_processes == 1 and device_map == "auto":
        #     self._device = torch.device(device)
        #     self.device_map = device_map
        # else:
        #     self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        #     self.device_map = f"cuda:{accelerator.local_process_index}"

        self.pretrained = pretrained
        # self._model = LlavaNextVideoForConditionalGeneration.from_pretrained(pretrained)
        # self._processor = LlavaNextVideoProcessor.from_pretrained(pretrained)
        self.prompt = "USER: <video>{}? ASSISTANT:"
        self.num_frames = num_frames
        assert num_frames == 32, "num_frames must be 8 https://github.com/huggingface/transformers/blob/bdb9106f247fca48a71eb384be25dbbd29b065a8/src/transformers/models/video_llava/modeling_video_llava.py#L379"
        self._config = self._model.config
        # self.model.eval()
        # self.model.tie_weights()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        # if accelerator.num_processes > 1:
        #     assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
        #     # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
        #     # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
        #     # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
        #     if accelerator.distributed_type == DistributedType.DEEPSPEED:
        #         kwargs = {
        #             "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
        #             "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
        #         }
        #         AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
        #         eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
        #     # if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
        #     #     self._model = accelerator.prepare(self.model)
        #     # else:
        #     #     self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
        #     self.accelerator = accelerator
        #     if self.accelerator.is_local_main_process:
        #         eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
        #     self._rank = self.accelerator.local_process_index
        #     self._world_size = self.accelerator.num_processes
        # elif accelerator.num_processes == 1 and device_map == "auto":
        #     eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
        #     self._rank = 0
        #     self._world_size = 1
        # else:
        #     eval_logger.info(f"Using single device: {self._device}")
        #     self.model.to(self._device)
        #     self._rank = 0
        #     self._world_size = 1
    
    # def _create_tokenizer(
    #     self,
    #     pretrained: Union[str, transformers.PreTrainedModel],
    #     revision: Optional[str] = "main",
    #     trust_remote_code: Optional[bool] = False,
    #     **kwargs,
    # ) -> None:
    #     # Get tokenizer based on 'pretrained'
    #     if isinstance(pretrained, str):
    #         model_name = pretrained
    #     else:
    #         # get the HF hub name via accessor on model
    #         model_name = self.model.name_or_path

    #     print(revision)
    #     self.processor = AutoProcessor.from_pretrained(
    #         model_name,
    #         revision=revision,
    #         trust_remote_code=trust_remote_code,
    #     )
    #     self.tokenizer = self.processor.tokenizer
    
    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    
    # @property
    # def tokenizer(self):
    #     return self._tokenizer

    
    # @property
    # def model(self):
    #     # returns the model, unwrapping it if using Accelerate
    #     if hasattr(self, "accelerator"):
    #         return self.accelerator.unwrap_model(self._model)
    #     else:
    #         return self._model

    
    # @property
    # def eot_token_id(self):
    #     # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
    #     return self.tokenizer.eos_token_id

    
    # @property
    # def max_length(self):
    #     return self._max_length

    
    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    
    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    
    # @property
    # def device(self):
    #     return self._device

    
    # @property
    # def rank(self):
    #     return self._rank

    
    # @property
    # def world_size(self):
    #     return self._world_size

    
    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def _model_multimodal_generate(self, inputs, max_length, stop, **generation_kwargs):
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")

        stopping_criteria = stop_sequences_criteria(
            self.tokenizer,
            stop,
            inputs["input_ids"].shape[1],
            inputs["input_ids"].shape[0],
        )
        return self.model.generate(
            **inputs,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **generation_kwargs,
        )
    
    # def _loglikelihood_tokens(
    #     self,
    #     requests: List[
    #         Tuple[Tuple[None, str, str], List[int], List[int], List[int]]
    #     ],  # TODO: update typehint to be correct
    #     disable_tqdm: bool = False,
    #     override_bs: int = None,
    # ) -> List[Tuple[float, bool]]:
    #     res = []

    #     # TODO: **improve multimodal collation.** We currently ignore image size when ordering docs. ideally we'd take them into account
    #     def _collate(req: Tuple[Tuple[str, str], List[int], List[int]]):
    #         """Defines the key for the sorted method"""
    #         # the negative sign on len(toks) sorts descending - this has a few advantages:
    #         # - time estimates will always be over not underestimates, which is more useful for planning
    #         # - to know the size of a batch when going through the list, you know the first one is always the batch
    #         #   padded context length. this is useful to simplify the batching logic and more importantly to make
    #         #   automatic adaptive batches much much easier to implement
    #         # - any OOMs will happen right away rather than near the end
    #         toks = req[1] + req[2]
    #         return -len(toks), tuple(toks)

    #     def _lookup_one_token_cont(req: Tuple[Tuple[str, str], List[int], List[int]]):
    #         """Defines the key to group and lookup one-token continuations"""
    #         # Use with group_by="contexts" (optional)"
    #         # allows for the creation of a lookup, so we can reuse logits in case of one-token continuations.
    #         # speeds up some multiple-choice tasks proportionally to the number of choices.
    #         # groups requests by context+continuation[:-1] and infer on one request/group.
    #         return req[-1] + req[-3] + req[-2][:-1]

    #     re_ord = Collator(
    #         requests,
    #         sort_fn=_collate,
    #         group_by="contexts"  # TODO: can't group-by just "contexts" any more, need to incorporate imgs
    #         if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
    #         and self.logits_cache
    #         else None,
    #         group_fn=_lookup_one_token_cont,
    #     )

    #     # automatic (variable) batch size detection for vectorization
    #     # pull longest context sample from request
    #     n_reordered_requests = len(re_ord)
    #     batch_size = (
    #         self.batch_size
    #         if self.batch_size != "auto"
    #         else override_bs
    #         if override_bs is not None
    #         else 0
    #     )
    #     batch_fn = (
    #         self._batch_scheduler
    #         if self.batch_size == "auto"
    #         and n_reordered_requests > 0
    #         and not override_bs
    #         else None
    #     )

    #     chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
    #     pbar = tqdm(
    #         total=len(requests),
    #         disable=(disable_tqdm or (self.rank != 0)),
    #         desc="Running loglikelihood requests with text+image input",
    #     )
    #     for chunk in chunks:
    #         imgs = []
    #         inps = []
    #         cont_toks_list = []
    #         inplens = []

    #         padding_len_inp = None
    #         # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
    #         # tensors, then we pack them together into a batch, call the model, and then pick it all apart
    #         # again because vectorizing is annoying

    #         for _, context_enc, continuation_enc, image_enc in chunk:
    #             # sanity check
    #             assert len(image_enc) > 0
    #             assert len(context_enc) > 0
    #             assert len(continuation_enc) > 0
    #             assert len(continuation_enc) <= self.max_length

    #             # how this all works (illustrated on a causal decoder-only setup):
    #             #          CTX      CONT
    #             # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
    #             # model  \               \
    #             # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
    #             # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

    #             # when too long to fit in context, truncate from the left
    #             # TODO: assuming that we won't handle enc-dec Vision2Seq models. Is that a safe assumption?
    #             inp = torch.tensor(
    #                 (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
    #                 dtype=torch.long,
    #                 device=self.device,
    #             )
    #             (inplen,) = inp.shape

    #             padding_len_inp = (
    #                 max(padding_len_inp, inplen)
    #                 if padding_len_inp is not None
    #                 else inplen
    #             )

    #             inps.append(inp)  # [1, inp_length]
    #             cont_toks_list.append(continuation_enc)
    #             inplens.append(inplen)

    #             imgs.append(image_enc)

    #         # create encoder attn mask and batched conts, if seq2seq
    #         call_kwargs = {}
    #         batched_inps = pad_and_concat(
    #             padding_len_inp, inps, padding_side="right"
    #         )  # [batch, padding_len_inp]
    #         # batch our examples' image inputs together
    #         batched_imgs = self._batch_images(
    #             imgs
    #         )  # TODO: fix/test for bs>1 case with differently-sized imgs!

    #         multi_logits = F.log_softmax(
    #             self._model_multimodal_call(batched_inps, batched_imgs, **call_kwargs),
    #             dim=-1,
    #         )  # [batch, padding_length (inp or cont), vocab]

    #         for (
    #             request_str,
    #             ctx_tokens,
    #             _,
    #             image_encs,
    #         ), logits, inplen, cont_toks in zip(
    #             chunk, multi_logits, inplens, cont_toks_list
    #         ):
    #             # Slice to original seq length
    #             contlen = len(cont_toks)
    #             # take only logits in the continuation
    #             # (discard context toks if decoder-only ; discard right-padding)
    #             # also discards + checks for "virtual tokens" in the causal LM's input window
    #             # from prompt/prefix tuning tokens, if applicable
    #             ctx_len = (
    #                 inplen + (logits.shape[0] - padding_len_inp)
    #                 if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
    #                 else None
    #             )
    #             logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
    #             logits = logits.unsqueeze(0)  # [1, seq, vocab]

    #             # Check if per-token argmax is exactly equal to continuation
    #             greedy_tokens = logits.argmax(dim=-1)

    #             # check for one-token continuation cache hits.
    #             # noop in case group_by != "contexts" or no cache hit and returns the
    #             # original args. Otherwise, expands the logits batch dimension and yields each
    #             # batch along with matching continuation tokens and prompt strings.
    #             # logits -> [1, seq, vocab]
    #             for request_str, cont_toks, logits in re_ord.get_cache(
    #                 req_str=request_str,
    #                 cxt_toks=ctx_tokens,
    #                 cont_toks=cont_toks,
    #                 logits=logits,
    #             ):
    #                 cont_toks = torch.tensor(
    #                     cont_toks, dtype=torch.long, device=self.device
    #                 ).unsqueeze(0)  # [1, seq]
    #                 max_equal = (greedy_tokens == cont_toks).all()

    #                 # Obtain log-probs at the corresponding continuation token indices
    #                 # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
    #                 logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
    #                     -1
    #                 )  # [1, seq]

    #                 # Answer: (log prob, is-exact-match)
    #                 answer = (float(logits.sum()), bool(max_equal))

    #                 res.append(answer)

    #                 self.cache_hook.add_partial(
    #                     "loglikelihood", request_str, answer
    #                 )  # TODO: choose convention for adding images into the cache key
    #                 pbar.update(1)

    #     pbar.close()

    #     return re_ord.get_original(res)

    
    # def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
    #     return super().loglikelihood(requests)


    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError(
            "model type `video_llava` does not support loglikelihood_rolling. Use 'hf' model type for text-only loglikelihood_rolling tasks ",
            "this is because we do not support measuring the loglikelihood a model assigns to an image.",
        )

    def loglikelihood(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[Tuple[float, bool]]:
        raise NotImplementedError(
            "'loglikelihood' requests for model type `video_llava` are not yet tested. This feature will be enabled when a loglikelihood-based multiple-choice VQA dataset is added!"
        )

    
    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list


    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]
        
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests with text+video input",
        )

        re_ords = Collator(
            [reg.args for reg in requests],
            _collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)
        for chunk in chunks:
            contexts, all_gen_kwargs, aux_arguments = zip(*chunk)

            if not isinstance(contexts, list):
                contexts = list(
                    contexts
                )
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                # add EOS token to stop sequences
                until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks
            max_ctx_len = self.max_length - max_gen_toks

            # only one video
            video = [arg["video"][0] for arg in aux_arguments][0]
            total_frames = len(video)
            indicies = np.arange(0, total_frames, total_frames / self.num_frames).astype(int)
            clip = video.get_batch(indicies).asnumpy()

            inputs = self.processor(text=contexts, videos=clip, return_tensors="pt")
            pixel_values_videos = inputs["pixel_values_videos"]

            if pixel_values_videos.shape[1] != self.num_frames:
                empty_frames = torch.zeros((1, self.num_frames - pixel_values_videos.shape[1], *pixel_values_videos.shape[2:]), dtype=pixel_values_videos.dtype)
                pixel_values_videos = torch.cat([pixel_values_videos, empty_frames], dim=1)
                inputs["pixel_values_videos"] = pixel_values_videos
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            context_enc = inputs["input_ids"]

            if "max_length" not in kwargs:
                kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

            generate_ids = self._model_multimodal_generate(inputs, stop=until,**kwargs)

            outputs = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].split("ASSISTANT:")[-1].strip()

            res.append(outputs)
            pbar.update(1)

        # res = []
        # pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # for contexts, gen_kwargs, video in [reg.args for reg in requests]:
        #     video = video['visual']
        #     total_frames = len(video)
        #     indicies = np.arange(0, total_frames, total_frames/self.num_frames).astype(int)
        #     clip = video.get_batch(indicies).asnumpy()
        #     inputs = self.processor(text=self.prompt.format(contexts), videos=clip, return_tensors="pt")
        #     pixel_values_videos = inputs["pixel_values_videos"]
        #     if pixel_values_videos.shape[1] != self.num_frames:
        #         empty_frames = torch.zeros((1, self.num_frames - pixel_values_videos.shape[1], *pixel_values_videos.shape[2:]), dtype=pixel_values_videos.dtype)
        #         pixel_values_videos = torch.cat([pixel_values_videos, empty_frames], dim=1)
        #         inputs["pixel_values_videos"] = pixel_values_videos
        #     inputs = {k: v.to(self.device) for k, v in inputs.items()}

        #     # if "max_new_tokens" not in gen_kwargs:
        #     #     gen_kwargs["max_new_tokens"] = 1024
        #     # if "temperature" not in gen_kwargs:
        #     #     gen_kwargs["temperature"] = 0
        #     # if "top_p" not in gen_kwargs:
        #     #     gen_kwargs["top_p"] = None
        #     # if "num_beams" not in gen_kwargs:
        #     #     gen_kwargs["num_beams"] = 1

        #     # generate_ids = self.model.generate(**inputs, max_new_tokens=gen_kwargs["max_new_tokens"], temperature=gen_kwargs["temperature"])

        #     generate_ids = self._model_multimodal_generate(self, inputs, max_length, stop, **generation_kwargs)
            
        #     outputs = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].split("ASSISTANT:")[-1].strip()
        #     res.append(outputs)
        #     pbar.update(1)
        return res
