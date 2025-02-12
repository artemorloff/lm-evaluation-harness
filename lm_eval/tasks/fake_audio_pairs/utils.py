import io
from typing import Any, Dict, List


def _doc_to_text(doc: Dict[str, Any]) -> str:
    """
    Helper function that processes the entire doc to form the input prompt only.

    :param doc: Dict[str, Any]
        one dataset sample as dictionary

    :return
        one string - the prompt to be passed into LM
    """

    # take the instruction and fill it with all doc["inputs"] data

    prompt = doc["instruction"].format(**doc["inputs"])

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
    audios = []
    for key in doc["inputs"]:
        if key.startswith("audio"):
            audios.append(doc["inputs"][key])

    return audios
