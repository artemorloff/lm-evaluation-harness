from . import (
    anthropic_llms,
    api_models,
    api_models_mera,
    dummy,
    gguf,
    hf_audiolm,
    hf_audiolm_mera,
    hf_steered,
    hf_videolm_mera,
    hf_vlms,
    hf_vlms_mera,
    huggingface,
    huggingface_mera,
    ibm_watsonx_ai,
    mamba_lm,
    nemo_lm,
    neuron_optimum,
    openai_completions,
    openai_completions_mera,
    optimum_ipex,
    optimum_lm,
    sglang_causallms,
    sglang_generate_API,
    textsynth,
    vllm_causallms,
    vllm_causallms_mera,
    vllm_vlms,
    vllm_vlms_mera
)


# TODO: implement __all__


try:
    # enable hf hub transfer if available
    import hf_transfer  # type: ignore # noqa
    import huggingface_hub.constants  # type: ignore

    huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
except ImportError:
    pass
