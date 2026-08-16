import json
import logging
from typing import Dict, List, Tuple

from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


eval_logger = logging.getLogger("lm-eval")


@register_model("jsonl_cached")
class JSONLCachedModel(LM):
    """
    Модель, которая берет готовые генерации из JSONL файла вместо реального вызова LM.
    Используется для оценки уже готовых генераций.
    """
    
    def __init__(self, jsonl_path: str = None, **kwargs):
        """
        Инициализация модели с JSONL файлом готовых генераций.
        
        Args:
            jsonl_path: путь к JSONL файлу с готовыми генерациями
        """
        super().__init__()
        
        if jsonl_path is None:
            raise ValueError("jsonl_path is required for JSONLCachedModel")
            
        self.jsonl_path = jsonl_path
        self.cache = {}
        self.misses = 0
        self._load_cache()
        
    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        """
        Создание модели из строки аргументов.
        Ожидаемый формат: jsonl_path=/path/to/file.jsonl
        """
        additional_config = {} if additional_config is None else additional_config
        
        # Парсим аргументы
        args = {}
        if arg_string:
            for arg in arg_string.split(','):
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    args[key.strip()] = value.strip()
        
        # Добавляем дополнительную конфигурацию
        args.update(additional_config)
        
        return cls(**args)
    
    def _load_cache(self):
        """Загружаем генерации из JSONL файла в кеш."""
        eval_logger.info(f"Loading cached generations from {self.jsonl_path}")
        
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line.strip():
                    try:
                        sample = json.loads(line)
                        doc = sample['doc']
                        resps = sample['resps']
                        
                        # Извлекаем генерации из структуры resps
                        # resps имеет структуру [[["gen1"], ["gen2"], ...]]
                        if isinstance(resps, list) and len(resps) > 0:
                            generations = resps[0]
                            if isinstance(generations, list):
                                # Каждая генерация может быть списком, извлекаем первый элемент
                                extracted_gens = []
                                for gen in generations:
                                    if isinstance(gen, list) and len(gen) > 0:
                                        extracted_gens.append(gen[0])
                                    else:
                                        extracted_gens.append(str(gen))
                                
                                # Ключей два, и первый из них — главный.
                                #
                                # generate_until ищет по request.args[0], то
                                # есть по КОНТЕКСТУ, который harness собрал при
                                # генерации: с few-shot примерами и, если был
                                # --apply_chat_template, в виде сериализованного
                                # чата. Ключ, восстановленный из doc, ничего
                                # этого не содержит и потому не совпадает с
                                # запросом ни разу: на реальном логе MERA
                                # (samples_rubin, 783 документа) так терялись
                                # все 100% попаданий, и каждый документ получал
                                # "# Cached generation not found".
                                #
                                # Сам контекст лежит в этом же файле, в
                                # arguments — это ровно та строка, которую
                                # harness подставит снова.
                                for key in self._recorded_contexts(sample):
                                    self.cache[key] = extracted_gens
                                # Ключ по документу остаётся как запасной: для
                                # логов без arguments и для вызова со своим
                                # контекстом.
                                cache_key = self._create_cache_key(doc)
                                self.cache.setdefault(cache_key, extracted_gens)

                    except json.JSONDecodeError as e:
                        eval_logger.warning(f"Failed to parse line {line_num}: {e}")
                    except KeyError as e:
                        eval_logger.warning(f"Missing key in line {line_num}: {e}")
                        
        eval_logger.info(f"Loaded {len(self.cache)} cached samples")
    
    @staticmethod
    def _recorded_contexts(sample: Dict) -> List[str]:
        """Контексты, с которыми документ реально генерировался.

        lm-eval пишет их в ``arguments`` в одном из двух видов:
        ``{"gen_args_0": {"arg_0": [context], "arg_1": {...}}}`` в свежих
        версиях и ``[[context, gen_kwargs]]`` в старых. Разбираем оба, а
        нераспознанное молча пропускаем — тогда сработает запасной ключ.
        """
        arguments = sample.get("arguments")
        contexts = []

        def take(value):
            if isinstance(value, str):
                contexts.append(value)
            elif isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
                contexts.append(value[0])

        if isinstance(arguments, dict):
            for entry in arguments.values():
                if isinstance(entry, dict):
                    take(entry.get("arg_0"))
                else:
                    take(entry)
        elif isinstance(arguments, (list, tuple)):
            for entry in arguments:
                take(entry)

        return contexts

    def _create_cache_key(self, doc: Dict) -> str:
        """
        Создаем уникальный ключ для документа.
        Используем instruction и inputs как основу для ключа.
        """
        instruction = doc.get('instruction', '')
        inputs = doc.get('inputs', {})

        # Формируем полный prompt как в doc_to_text. Подстановка делается по
        # данным задачи, а они могут не совпасть с плейсхолдерами инструкции —
        # тогда берём инструкцию как есть, вместо того чтобы уронить загрузку
        # всего файла на одном документе.
        try:
            if isinstance(inputs, dict):
                prompt = instruction.format(**inputs)
            else:
                prompt = instruction.format(inputs=inputs) if '{inputs}' in instruction else instruction
        except (KeyError, IndexError, ValueError):
            prompt = instruction

        return prompt.strip()
    
    def _get_cached_generations(self, context: str, num_generations: int = 1) -> List[str]:
        """Получаем готовые генерации для данного контекста."""
        if context in self.cache and self.cache[context]:
            generations = self.cache[context]
            # Возвращаем нужное количество генераций
            if len(generations) >= num_generations:
                return generations[:num_generations]
            else:
                # Если генераций меньше чем нужно, дублируем последние
                result = generations.copy()
                while len(result) < num_generations:
                    result.extend(generations)
                return result[:num_generations]
        else:
            eval_logger.warning(f"No cached generation found for context: {context[:100]}...")
            return ["# Cached generation not found"] * num_generations
    
    def loglikelihood(self, requests, disable_tqdm: bool = False) -> List[Tuple[float, bool]]:
        """
        Для loglikelihood возвращаем фиктивные значения, так как мы работаем с готовыми генерациями.
        """
        res = []
        for request in tqdm(requests, disable=disable_tqdm, desc="Processing loglikelihood requests"):
            # Возвращаем нейтральные значения
            res.append((0.0, True))
        return res
    
    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> List[float]:
        """
        Для loglikelihood_rolling возвращаем фиктивные значения.
        """
        res = []
        for request in tqdm(requests, disable=disable_tqdm, desc="Processing loglikelihood_rolling requests"):
            res.append(0.0)
        return res
    
    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        """
        Основная функция - возвращаем готовые генерации из кеша.
        """
        res = []
        context_counters = {}  # Счетчик для отслеживания какая генерация нужна для каждого контекста
        
        for request in tqdm(requests, disable=disable_tqdm, desc="Processing generation requests"):
            context = request.args[0]  # Извлекаем контекст из запроса
            
            # Отслеживаем какая по счету генерация нужна для этого контекста
            if context not in context_counters:
                context_counters[context] = 0
            # Получаем все генерации для контекста
            if context in self.cache and self.cache[context]:
                generations = self.cache[context]
                # Берем генерацию по кругу если запросов больше чем генераций
                generation_idx = context_counters[context] % len(generations)
                generation = generations[generation_idx]
                context_counters[context] += 1
            else:
                self.misses += 1
                eval_logger.warning(f"No cached generation found for context: {context[:100]}...")
                generation = "# Cached generation not found"

            res.append(generation)

        # Промах подставляет текст-заглушку, а метрики посчитают её как обычный
        # ответ модели — то есть неверный. Поодиночке это видно только в
        # warning'ах, которые тонут в выводе, поэтому итог сообщаем отдельно.
        if self.misses:
            eval_logger.error(
                "jsonl_cached: %d of %d requests had no cached generation and "
                "were answered with a placeholder — the metrics below count "
                "those as model answers. Check that %s was produced by the same "
                "task and the same prompt settings (few-shot, chat template).",
                self.misses,
                len(requests),
                self.jsonl_path,
            )

        return res
    
    @property
    def tokenizer_name(self) -> str:
        """Возвращаем имя для кеширования."""
        return f"jsonl_cached_{self.jsonl_path}"
    
    def chat_template(self, chat_template=False) -> str:
        """
        Возвращает шаблон чата (заглушка).
        
        Args:
            chat_template: параметр шаблона чата
            
        Returns:
            пустая строка как заглушка
        """
        return ""
    
    def apply_chat_template(self, chat_history, **kwargs) -> str:
        """
        Применяет шаблон чата к истории (заглушка).
        
        Args:
            chat_history: история чата
            
        Returns:
            строка с примененным шаблоном (просто берем последнее сообщение пользователя)
        """
        if not chat_history:
            return ""
        
        # Простая заглушка - просто берем последнее сообщение пользователя
        for message in reversed(chat_history):
            if message.get("role") == "user":
                return message.get("content", "")
        
        return "" 