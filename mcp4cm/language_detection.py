from collections import defaultdict


from mcp4cm.base import Dataset, Model
from mcp4cm.bpmn.dataloading.bpmn_dataset import BPMNDataset
from mcp4cm._language_detector import _get_text_language, _initialize_language_detector_seed
from mcp4cm.bpmn.language_detection import filter_models_by_language as filter_bpmn_models_by_language
from mcp4cm.bpmn.data_extraction import extract_dataset_languages as extract_bpmn_dataset_languages


def detect_model_language(model: Model, key: str = 'names', empty_name: str | None = None, force_detection: bool = False) -> str:
    """
    Process a single model to detect its language.
    
    This function uses langdetect to identify the language of the text content
    in a Model. Optionally filters out occurrences of empty names from the text.
    
    Args:
        model (Model): The model to process for language detection.
        key (str, optional): The name of the property where the text content of the model is stored.
            Defaults to 'names'.
        empty_name (str, optional): The string which represents an empty name. Set to None if no text should be filtered before detection. Defaults to None.
        force_detection (bool, optional): If True, models which already have the language property set will be recomputed.
         Defaults to False.

    Returns:
        str: ISO 639-1 language code (e.g., 'en' for English, 'fr' for French),
             or None if the model has no text content or detection fails.
    """
    if model.language and not force_detection:
        return model.language

    text = model.get_text(key, empty_name=empty_name)
    return _get_text_language(text)


def extract_dataset_languages(dataset: Dataset, text_key: str = 'names', empty_name: str | None = None, override: bool = False) -> None:
    """
    Extracts the language of each model in the dataset and saves in the 'language' property.

    This function uses langdetect to identify the language of the text content in a Model.
    Empty names are filtered to not influence the results of the language detection.

    Args:
        dataset (Dataset): The dataset containing models.
        text_key (str, optional): The name of the property where the text content of the model is stored.
         Defaults to 'names'.
        empty_name (str, optional): The string which represents an empty name.
            Set to None if no text should be filtered before detection. Defaults to None.
        override (bool, optional): If True, models which already have the language property set will be recomputed.
         Defaults to False.
    """

    _initialize_language_detector_seed(seed=0)
    if isinstance(dataset, BPMNDataset):
        extract_bpmn_dataset_languages(dataset, text_key=text_key, empty_name=empty_name, override=override)
        return

    for model in dataset:
        if not model.language or override:
            model.language = detect_model_language(model, key=text_key, empty_name=empty_name, force_detection=override)



def get_models_by_language(dataset: Dataset, print_counts: bool = True) -> dict:
    """
    Call extract_dataset_languages() before this function is called to ensure the languages are known.

    Args:
        dataset (Dataset): The dataset containing models.
        print_counts (bool): Whether the amount of models per language should be printed.

    Returns:
        dict: A dictionary where keys are language codes (e.g., 'en', 'fr') and
            values are lists of Model objects in that language.

    Example:
        >>> languages = get_models_by_language(dataset)
        >>> print(f"Found {len(languages['en'])} English models")
    """

    language_dict = defaultdict(list)

    for model in dataset:
        language_dict[model.language].append(model)

    if print_counts:
        print("Language Distribution Across Models:")
        for lang, models in language_dict.items():
            print(f"Language: {lang}, Count: {len(models)}")

    return language_dict


def extract_non_english_models(dataset: Dataset, empty_name: str | None = None) -> Dataset:
    """
    Extract non-English models from the dataset.
    
    This function creates a new dataset containing only models whose text
    content is not in English. This is useful for filtering out non-English
    models for language-specific analysis or cleaning.
    
    Args:
        dataset (UMLDataset): The dataset containing UML models.
    
    Returns:
        UMLDataset: A new dataset containing only non-English models.
        
    Example:
        >>> non_english = extract_non_english_models(dataset)
        >>> print(f"Found {len(non_english.models)} non-English models")
    """
    non_english_models = []

    for model in dataset:
        if model.model_txt is None:
            continue
        lang = detect_model_language(model, empty_name=empty_name)
        if lang and lang != 'en':
            non_english_models.append(model)

    return Dataset(name=dataset.name, models=non_english_models)


def filter_models_by_language(dataset: Dataset,
                              language: str,
                              key: str = 'names',
                              empty_name: str | None = None) -> Dataset:
    """
    Filter models in the dataset by a specific language.
    
    This function returns a new dataset containing only models that match
    the specified language code.
    
    Args:
        dataset (Dataset): The dataset containing models.
        language (str): The ISO 639-1 language code to filter by (e.g., 'en', 'fr').
    
    Returns:
        Dataset: A new dataset containing only models in the specified language.
        
    Example:
        >>> english_models = filter_models_by_language(dataset, 'en')
        >>> print(f"Found {len(english_models.models)} English models")
    """
    if isinstance(dataset, BPMNDataset):
        return filter_bpmn_models_by_language(dataset, language, key, empty_name)

    filtered_models = [model for model in dataset if
                       detect_model_language(model, key, empty_name=empty_name, force_detection=False) == language]
    return Dataset(name=dataset.name, models=filtered_models)
