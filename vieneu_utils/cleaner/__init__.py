import re
from .num2vi import n2w, n2w_single

from .numerical import normalize_number_vi
from .datestime import normalize_date, normalize_time
from .text_norm import normalize_others

def clean_vietnamese_text(text):
    mask_map = {}
    import string
    
    def protect(match):
        idx = len(mask_map)
        mask = "MASK" + "".join([string.ascii_uppercase[int(d)] for d in str(idx).zfill(4)]) + "MASK"
        mask_map[mask] = match.group(0)
        return mask
    
    text = re.sub(r'___PROTECTED_EN_TAG_\d+___', protect, text)
    
    def _expand_float(m):
        int_part = n2w(m.group(1))
        dec_part = n2w(m.group(2))
        return f"{int_part} phẩy {dec_part}"
    text = re.sub(r'\b(\d+),(\d+)\b', _expand_float, text)
    
    def _strip_dot_sep(m):
        return m.group(0).replace('.', '')
    text = re.sub(r'\b\d+(?:\.\d{3})+\b', _strip_dot_sep, text)
    
    text = re.sub(r'(\d+)\s+[–\-~]\s+(\d+)', r'\1 đến \2', text)
    text = re.sub(r'(\d+)[–~](\d+)', r'\1 đến \2', text)
    text = re.sub(r'\s*(?:->|=>)\s*', ' sang ', text)

    text = normalize_date(text)
    text = normalize_time(text)
    
    text = normalize_others(text)
    text = normalize_number_vi(text)

    for mask, original in mask_map.items():
        text = text.replace(mask, original)
        text = text.replace(mask.lower(), original)
        
    return text
