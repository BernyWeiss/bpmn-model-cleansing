from langdetect import detect, LangDetectException, DetectorFactory


def _initialize_language_detector_seed(seed: int) -> None:
    DetectorFactory.seed = 0  # Set seed for reproducibility


def _get_text_language(text: str) -> str:
    if text and text.strip():  # Ensure it's not empty or whitespace
        try:
            return detect(text)
        except LangDetectException:
            return None
    return None
