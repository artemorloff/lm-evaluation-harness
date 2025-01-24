import logging
from collections import defaultdict
from statistics import mean, median
from typing import Dict, List, Tuple, Union

from lm_eval.utils import simple_parse_args_string


logger = logging.getLogger("lm-eval")

##############################################################################
# Глобальный реестр для статистики обрезки. 
##############################################################################
_TRUNC_STATS = defaultdict(lambda: {
    "total_samples": 0,      # общее число обработанных сэмплов
    "truncated_samples": 0,  # сколько реально пришлось обрезать
    "orig_lengths": [],      # длины (токенов/символов) до обрезки
    "trunc_lengths": [],     # длины после обрезки
    "cut_amounts": []        # на сколько укоротили (orig_len - trunc_len)
})


def process_truncation_args(args: Union[str, Dict[str, Union[str, bool, int]]]) -> Dict[str, Union[str, bool, int]]:
    """
    Преобразует строку (формата: "how=default,on=tokens,side=left,...") или словарь
    c настройками обрезки в единый словарь с дефолтными значениями.

    Параметры в выходном словаре:
    -----------------------------
    how: str             - Режим обрезки ('no', 'default', 'fewshots', 'user', 'transformers')
    on: str              - 'tokens' или 'symbols'
    side: str            - 'left' или 'right'
    keep_first: bool     - Нужно ли сохранять первый фьюшот целиком (актуально для how='fewshots')
    max_symbols: int     - Лимит по символам (on='symbols')
    max_new_symbols: int - Резерв по символам под generate
    max_length: int      - Лимит по токенам (on='tokens')
    max_new_tokens: int  - Резерв по токенам под generate
    """
    default_args = {
        "how": "no",
        "on": "tokens",
        "side": "left",
        "keep_first": False,
        "max_symbols": 2048,
        "max_new_symbols": 256,
        "max_length": 2048,
        "max_new_tokens": 256,
    }
    if isinstance(args, str):
        parsed = simple_parse_args_string(args)
        default_args.update(parsed)
    elif isinstance(args, dict):
        default_args.update(args)
    return default_args


def tokenize_sequence(
    seq: Union[str, List, Dict],
    model,
    add_special_tokens: bool = False,
    do_symbol: bool = False
) -> List[int]:
    """
    Превращает входную последовательность seq в список "токенов".
      - Если do_symbol=True, 1 символ = 1 "токен" .
      - Иначе пытаемся вызвать модельную токенизацию:
         1) model.tokenizer(...)
         2) model.tokenize(...)
         3) fallback: seq.split()

    seq:  может быть строкой или списком (например, чат-история).
    model: объект модели, у которой может быть tokenizer, либо метод tokenize().
    """
    # Если включен режим "symbols", разбиваем по символам
    if do_symbol:
        if isinstance(seq, str):
            return list(range(len(seq)))  # просто range от 0..длина
        elif isinstance(seq, list):
            merged = _convert_list_to_string(seq)
            return list(range(len(merged)))
        else:
            return []

    # Если do_symbol=False, пробуем реальную токенизацию
    text = seq if isinstance(seq, str) else _convert_list_to_string(seq)

    tokenizer = getattr(model, "tokenizer", None)
    if callable(tokenizer):
        enc = tokenizer(text, add_special_tokens=add_special_tokens)
        if "input_ids" in enc:
            return enc["input_ids"]

    if hasattr(model, "tokenize") and callable(model.tokenize):
        return model.tokenize(text)

    # Если ничего не получилось 
    return text.split()


def apply_chat_template(
    seq: Union[List, str],
    model,
    add_generation_prompt: bool = False,
    do_tokenize: bool = False
) -> Union[str, List[int]]:
    """
    Применяет (при наличии) chat-шаблон модели к seq.
    Если do_tokenize=True, возвращает список токенов, иначе готовую строку.
    """
    if hasattr(model, "chat_template") and callable(model.chat_template):
        return model.chat_template(seq, add_generation_prompt=add_generation_prompt, tokenize=do_tokenize)
    else:
        text = seq if isinstance(seq, str) else _convert_list_to_string(seq)
        if do_tokenize:
            return tokenize_sequence(text, model, do_symbol=False)
        return text


def truncate_and_chat_template(
    request,
    lm,
    chat_template,
    truncation_args: Dict[str, Union[str, bool, int]],
    first_system: bool,
    task_name: str = "unknown_task"
) -> Tuple:
    """
    Главная функция обрезки + (опциональной) chat-шаблонизации каждого запроса (Instance).
    Возвращает (обновлённый_request, status), где status — строка с кодом (например, 'truncated_default').

    - request:   экземпляр Instance (из evaluator), у которого есть .arguments
    - lm:        модель (например, HF-модель, vLLM-модель и т.д.)
    - chat_template: функц. шаблона (можно передать None, если не нужно)
    - truncation_args: словарь, полученный из process_truncation_args
    - first_system:   флаг о том, что первая реплика — system (для чатов, если нужно)
    - task_name:      имя задачи для логгирования статистики
    """
    how = truncation_args.get("how", "no")
    on = truncation_args.get("on", "tokens")
    side = truncation_args.get("side", "left")
    keep_first = truncation_args.get("keep_first", False)
    max_symbols = truncation_args.get("max_symbols", 2048)
    max_new_symbols = truncation_args.get("max_new_symbols", 256)
    max_length = truncation_args.get("max_length", 2048)
    max_new_tokens = truncation_args.get("max_new_tokens", 256)

    # Распаковываем аргументы
    req_type = request.request_type
    args = request.arguments
    if len(args) == 1:
        ctx, cont = args[0], ""
    else:
        ctx, cont = args[0], args[1]

    # Определяем, нужно ли резервировать под генерацию
    is_gen = (req_type == "generate_until")
    # Определяем реальный max_context_len
    if on == "tokens":
        max_context_len = max_length - (max_new_tokens if is_gen else 0)
    else:  # 'symbols'
        max_context_len = max_symbols - (max_new_symbols if is_gen else 0)

    # Меряем длину оригинала
    do_symbol = (on == "symbols")
    original_tokens = tokenize_sequence(ctx, lm, add_special_tokens=False, do_symbol=do_symbol)
    original_len = len(original_tokens)

    # Если не остаётся места под prompt
    if max_context_len < 1:
        # Обнулим контекст
        request.arguments = ("", cont) if len(args) > 1 else ("",)
        register_truncation_stats(task_name, original_len, 0)  # зафиксируем
        return request, f"error_{how}_no_space"

    # Если how=='no' — не обрезаем
    if how == "no":
        register_truncation_stats(task_name, original_len, original_len)
        return request, "no_truncation"

    # Если how=='transformers' — встроенная обрезка
    if how == "transformers":
        truncated_ctx = _transformers_truncate(ctx, lm, max_context_len, on)
        new_tokens = tokenize_sequence(truncated_ctx, lm, do_symbol=do_symbol)
        register_truncation_stats(task_name, original_len, len(new_tokens))
        request.arguments = (truncated_ctx, cont) if len(args) > 1 else (truncated_ctx,)
        return request, "truncated_transformers"

    # Иначе — ручные стратегии
    if original_len <= max_context_len:
        # Помещается без обрезки
        register_truncation_stats(task_name, original_len, original_len)
        return request, f"no_truncation_{how}"

    if how == "default":
        new_ctx_tokens = _truncate_list_side(original_tokens, max_context_len, side=side)
        new_ctx = _untokenize_sequence(new_ctx_tokens, ctx, lm, do_symbol)
        register_truncation_stats(task_name, original_len, len(new_ctx_tokens))
        request.arguments = (new_ctx, cont) if len(args) > 1 else (new_ctx,)
        return request, "truncated_default"

    elif how == "user":
        # Режем справа, оставляя (max_context_len) "токенов"/символов
        cut_count = original_len - max_context_len
        new_ctx_tokens = original_tokens[:-cut_count]
        new_ctx = _untokenize_sequence(new_ctx_tokens, ctx, lm, do_symbol)
        register_truncation_stats(task_name, original_len, len(new_ctx_tokens))
        request.arguments = (new_ctx, cont) if len(args) > 1 else (new_ctx,)
        return request, "truncated_user"

    elif how == "fewshots":
        new_ctx = _drop_fewshots(ctx, lm, max_context_len, on, side, keep_first)
        new_len = len(tokenize_sequence(new_ctx, lm, do_symbol=do_symbol))
        register_truncation_stats(task_name, original_len, new_len)
        request.arguments = (new_ctx, cont) if len(args) > 1 else (new_ctx,)
        return request, "truncated_fewshots"

    # Если режим не поддерживается
    register_truncation_stats(task_name, original_len, original_len)
    return request, f"not_implemented_{how}"


##############################################################################
# Вспомогательные функции для разных стратегий
##############################################################################

def _transformers_truncate(text, model, max_len: int, on: str) -> str:
    """Адаптивная обрезка через встроенную truncation у HF-токенизатора.
       Если on='symbols', режем напрямую по символам.
    """
    if on == "symbols":
        return text[:max_len]
    tokenizer = getattr(model, "tokenizer", None)
    if not tokenizer:
        # fallback
        return text[:max_len]

    encoded = tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False
    )
    return tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)


def _truncate_list_side(tokens: List, max_len: int, side: str = "left") -> List:
    """
    Обрезаем список (токенов/символов) до размера max_len:
       - side='left' => берём последние max_len
       - side='right' => первые max_len
    """
    if len(tokens) <= max_len:
        return tokens
    if side == "left":
        return tokens[-max_len:]
    return tokens[:max_len]


def _untokenize_sequence(
    tokens: List,
    original_ctx: Union[str, List],
    model,
    do_symbol: bool
) -> str:
    """
    Превращает список tokens обратно в строку (если do_symbol=False) через model.tokenizer.decode
    или просто обрезку (если do_symbol=True).
    """
    if do_symbol:
        # tokens — это диапазоны [0..n). Нужно вернуть кусок original_ctx
        if isinstance(original_ctx, str):
            return original_ctx[: len(tokens)]
        # fallback: склеим
        full_text = _convert_list_to_string(original_ctx)
        return full_text[: len(tokens)]
    else:
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer:
            return tokenizer.decode(tokens, skip_special_tokens=True)
        else:
            # fallback: склеим (если это список строк)
            if len(tokens) and isinstance(tokens[0], str):
                return " ".join(tokens)
            return str(tokens)


def _drop_fewshots(
    ctx: str,
    model,
    max_context_len: int,
    on: str,
    side: str,
    keep_first: bool
) -> str:
    """
    Примерная функция для "как бы" удаления целых фьюшотов.
    Упрощённо ищем в ctx разделитель 'FS_DELIM'. Убираем блоки целиком, пока не влезет.
    """
    do_symbol = (on == "symbols")
    parts = ctx.split("FS_DELIM")
    def length_of(text):
        return len(tokenize_sequence(text, model, do_symbol=do_symbol))

    # Если и так влезает, возвращаем как есть
    if length_of(ctx) <= max_context_len:
        return ctx

    remain = parts[:]
    while len(remain) > 1:
        joined = "FS_DELIM".join(remain)
        if length_of(joined) <= max_context_len:
            break
        if side == "right":
            if keep_first and len(remain) == 2:
                # если осталось два блока: 1 фьюшот + doc
                break
            remain.pop()
        else:  # side=='left'
            if keep_first and len(remain) > 1:
                # сохраняем первый блок
                if len(remain) == 2:
                    break
                remain.pop(1)
            else:
                remain.pop(0)
    return "FS_DELIM".join(remain)


def _convert_list_to_string(seq: Union[List, str]) -> str:
    """
    Утилита: склеить список (или строку) в строку. Если элементы - dict c полем 'content', тоже добавляем.
    """
    if isinstance(seq, str):
        return seq
    chunks = []
    for item in seq:
        if isinstance(item, dict) and "content" in item:
            c = item["content"]
            if isinstance(c, list):
                chunks.append(" ".join(str(x) for x in c))
            else:
                chunks.append(str(c))
        else:
            chunks.append(str(item))
    return " ".join(chunks)


##############################################################################
# Методы для ведения статистики
##############################################################################

def register_truncation_stats(
    task_name: str,
    orig_len: int,
    trunc_len: int
):
    """
    Фиксируем в глобальном реестре статистику обрезки:
    - orig_len: длина (токенов/символов) до обрезки
    - trunc_len: после
    """
    d = _TRUNC_STATS[task_name]
    d["total_samples"] += 1
    d["orig_lengths"].append(orig_len)
    d["trunc_lengths"].append(trunc_len)
    d["cut_amounts"].append(orig_len - trunc_len)
    if trunc_len < orig_len:
        d["truncated_samples"] += 1

def print_truncation_stats():
    """
    Вывести итоги по всем задачам, которые регистрировались через register_truncation_stats.
    """
    for tname, data in _TRUNC_STATS.items():
        total = data["total_samples"]
        truncated = data["truncated_samples"]
        logger.info(f"[Truncation Stats for task='{tname}']")
        logger.info(f"  total_samples: {total}")
        logger.info(f"  truncated_samples: {truncated} ({100.0*truncated/total:.1f}%)")
        if total > 0:
            o = data["orig_lengths"]
            t = data["trunc_lengths"]
            c = data["cut_amounts"]
            logger.info("  original_length:   min=%.1f, max=%.1f, mean=%.1f, median=%.1f" % (
                min(o), max(o), mean(o), median(o)
            ))
            logger.info("  truncated_length:  min=%.1f, max=%.1f, mean=%.1f, median=%.1f" % (
                min(t), max(t), mean(t), median(t)
            ))
            logger.info("  cut_amount:        min=%.1f, max=%.1f, mean=%.1f, median=%.1f" % (
                min(c), max(c), mean(c), median(c)
            ))
        logger.info("")
