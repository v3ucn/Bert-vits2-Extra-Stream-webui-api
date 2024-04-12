from text.symbols import *

_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def cleaned_text_to_sequence(cleaned_text, tones, language):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    phones = [_symbol_to_id[symbol] for symbol in cleaned_text]
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    lang_id = language_id_map[language]
    lang_ids = [lang_id for i in phones]
    return phones, tones, lang_ids


def get_bert(norm_text, word2ph, language, device, style_text=None, style_weight=0.7):
    from .chinese_bert import get_bert_feature as zh_bert
    bert = zh_bert(
        norm_text, word2ph, device, style_text, style_weight
    )
    return bert

def init_openjtalk():
    import platform

    if platform.platform() == "Linux":
        import pyopenjtalk

        pyopenjtalk.g2p("こんにちは，世界。")


init_openjtalk()
