import re
import ftfy
from langdetect import detect, LangDetectException


SEO_PATTERNS = [
    r"click here", r"buy now", r"free shipping",
    r"\d+% off", r"limited time", r"copyright",
    r"all rights reserved", r"stock photo",
    r"getty images", r"shutterstock", r"alamy",
    r"royalty.?free", r"editorial use",
]


def clean_caption(
    text       : str,
    min_words  : int   = 3,
    max_words  : int   = 50,
    lang_check : bool  = True,
) -> str | None:
    """
    Limpa e valida uma legenda do COYO para uso em treino contrastivo.

    Returns None se a amostra deve ser descartada.

    Parameters
    ----------
    text:
        Legenda bruta do dataset.
    min_words:
        Mínimo de palavras após limpeza.
    max_words:
        Máximo de palavras — trunca se ultrapassar.
    lang_check:
        Se True, descarta textos não-ingleses.
    """
    if not text or not isinstance(text, str):
        return None

    # 1. Corrige encoding quebrado
    text = ftfy.fix_text(text)

    # 2. Remove HTML tags e entities
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&\w+;', ' ', text)

    # 3. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # 4. Remove números longos isolados
    text = re.sub(r'\b\d{4,}\b', ' ', text)

    # 5. Remove caracteres especiais excessivos
    text = re.sub(r'[^\w\s\'\-\.,!?]', ' ', text)

    # 6. Normaliza espaços
    text = re.sub(r'\s+', ' ', text).strip()

    # 7. Filtra por tamanho
    words = text.split()
    if len(words) < min_words:
        return None
    if len(words) > max_words:
        text  = ' '.join(words[:max_words])
        words = words[:max_words]

    # 8. Filtra lixo comum
    blacklist = {
        "null", "undefined", "untitled", "image", "photo",
        "img", "none", "n/a", "na", "no description",
        "no image", "placeholder",
    }
    if text.lower().strip() in blacklist:
        return None

    # 9. Filtra textos sem conteúdo semântico
    meaningful_words = [w for w in words if len(w) > 2 and w.isalpha()]
    if len(meaningful_words) < min_words:
        return None

    # 10. Filtra repetição excessiva de tokens
    #unique_ratio = len(set(words)) / len(words)
    #if unique_ratio < 0.5:
    #    return None

    # 11. Filtra ruído de SEO / watermarks
    text_lower = text.lower()
    if any(re.search(p, text_lower) for p in SEO_PATTERNS):
        return None

    # 12. Filtra caracteres especiais excessivos
    special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
    if special_ratio > 0.3:
        return None

    # 13. Detecção de idioma (opcional — adiciona ~5ms por sample)
    #if lang_check:
    #    try:
    #        if detect(text) != "en":
    #            return None
    #    except LangDetectException:
    #        return None

    return text