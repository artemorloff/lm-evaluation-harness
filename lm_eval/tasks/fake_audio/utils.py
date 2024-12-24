import io
import os
import soundfile as sf
import time
from typing import Any, Dict, List

HF_HOME = os.getenv("HF_HOME", "~/.cache/huggingface/")
HF_TASK_CACHE_DIR = "fake_audio_data"
CACHE_PATH = os.path.join(os.path.expanduser(HF_HOME), HF_TASK_CACHE_DIR)


def _doc_to_text(doc: Dict[str, Any]) -> str:
    """
    Helper function that processes the entire doc to form the input prompt only.

    :param doc: Dict[str, Any]
        one dataset sample as dictionary

    :return
        one string - the prompt to be passed into LM
    """

    # take the instruction and fill it with all doc["inputs"] data

    prompt = doc["inputs"]["question"]

    return prompt


def doc_to_text(doc: Dict[str, Any]) -> str:
    """
    Get the prompt for a given document.
    This function is used by code to form prompt for each sample.

    :param doc: Dict[str, Any]
        one dataset sample as dictionary

    :return
        one string - the prompt to be passed into LM
    """

    prompt = _doc_to_text(doc)

    return prompt


def doc_to_audio(doc: Dict[str, Any]) -> List[str]:
    """
    Process audios.
    :param doc: Dict[str, Any]
        one dataset sample as dictionary
    :return
        list of audios
    """

    sr = doc["inputs"]["audio"]["sampling_rate"]
    audio_name = doc["inputs"]["audio"]["path"]
    data = doc["inputs"]["audio"]["array"]
    audio_path_on_disk = os.path.join(CACHE_PATH, audio_name)
    
    if not os.path.exists(CACHE_PATH):
        os.makedirs(CACHE_PATH)
    sf.write(audio_path_on_disk, data, sr)
    
    doc["inputs"]["audio"]["path"] = audio_path_on_disk
    audios = [doc["inputs"]["audio"]]
    return audios
