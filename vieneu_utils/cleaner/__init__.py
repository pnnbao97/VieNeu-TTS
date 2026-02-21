import re
from .num2vi import n2w, n2w_single

from .numerical import normalize_number_vi
from .datestime import normalize_date, normalize_time
from .text_norm import normalize_others, expand_measurement, expand_currency, expand_compound_units, expand_abbreviations, expand_standalone_letters

def clean_vietnamese_text(text):
    mask_map = {}
    import string
    
    def protect(match):
        idx = len(mask_map)
        mask = "MASK" + "".join([string.ascii_uppercase[int(d)] for d in str(idx).zfill(4)]) + "MASK"
        mask_map[mask] = match.group(0)
        return mask
    
    text = re.sub(r'___PROTECTED_EN_TAG_\d+___', protect, text)
    
    # Handle common abbreviations early to avoid unit conflicts
    text = expand_abbreviations(text)
    
    text = normalize_date(text)
    text = normalize_time(text)

    text = re.sub(r'(\d+(?:,\d+)?)\s*[–\-—~]\s*(\d+(?:,\d+)?)', r'\1 đến \2', text)
    text = re.sub(r'\s*(?:->|=>)\s*', ' sang ', text)

    # Expand measurements and currencies BEFORE general floats
    text = expand_compound_units(text)
    text = expand_measurement(text)
    text = expand_currency(text)

    def _expand_float(m):
        int_part = n2w(m.group(1))
        dec_part = n2w(m.group(2))
        res = f"{int_part} phẩy {dec_part}"
        if m.group(3):
            res += " phần trăm"
        return f" {res} "
    text = re.sub(r'\b(\d+),(\d+)(%)?', _expand_float, text)
    
    def _strip_dot_sep(m):
        return m.group(0).replace('.', '')
    text = re.sub(r'\b\d+(?:\.\d{3})+\b', _strip_dot_sep, text)
    
    text = normalize_others(text)
    text = normalize_number_vi(text)
    
    # Finally expand standalone letters to catch initials like "M."
    text = expand_standalone_letters(text)

    # Collapse redundant punctuation and whitespace
    # 1. Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # 2. Collapse consecutive commas and handle comma-punctuation pairs
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r',\s*([.!?;])', r'\1', text)
    # 3. Handle redundant spaces before punctuation
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    # 4. Remove leading/trailing commas if they end up at the start/end of sentence parts
    text = text.strip().strip(',')

    for mask, original in mask_map.items():
        text = text.replace(mask, original)
        text = text.replace(mask.lower(), original)
        
    return text
