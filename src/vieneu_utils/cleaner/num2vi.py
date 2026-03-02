from typing import List, Optional

units = {
    '0': 'không',
    '1': 'một',
    '2': 'hai',
    '3': 'ba',
    '4': 'bốn',
    '5': 'năm',
    '6': 'sáu',
    '7': 'bảy',
    '8': 'tám',
    '9': 'chín',
}

def chunks(lst: List, n: int) -> List[List]:
    """Split list into chunks of size n."""
    return [lst[i : i + n] for i in range(0, len(lst), n)]

def n2w_units(numbers: str) -> str:
    """Convert single digit to word."""
    if not numbers:
        raise ValueError('Số rỗng!!')
    if len(numbers) > 1:
        raise ValueError('Số vượt quá giá trị của hàng đơn vị!')
    return units[numbers]

def pre_process_n2w(number: str) -> Optional[str]:
    """Clean input number string."""
    clean = str(number).translate(str.maketrans('', '', ' -.,'))
    return clean if clean.isdigit() else None

def process_n2w_single(numbers: str) -> str:
    """Convert each digit independently to word."""
    return ' '.join(units[d] for d in numbers if d in units)

def n2w_hundreds(numbers: str) -> str:
    """Convert 1-3 digit number to Vietnamese words."""
    if not numbers or numbers == '000':
        return ""
    
    # Pad to 3 digits
    n = numbers.zfill(3)
    h_digit, t_digit, u_digit = n[0], n[1], n[2]

    res = []

    # Hundreds
    if h_digit != '0':
        res.append(units[h_digit] + " trăm")
    elif len(numbers) == 3: # Case like 0xx in a larger number
        res.append("không trăm")

    # Tens
    if t_digit == '0':
        if u_digit != '0' and (h_digit != '0' or len(numbers) == 3):
            res.append("lẻ")
    elif t_digit == '1':
        res.append("mười")
    else:
        res.append(units[t_digit] + " mươi")

    # Units
    if u_digit != '0':
        if u_digit == '1' and t_digit not in ('0', '1'):
            res.append("mốt")
        elif u_digit == '5' and (t_digit != '0' or (h_digit != '0' or len(numbers) == 3)):
            res.append("lăm")
        elif u_digit == '4' and t_digit not in ('0', '1'):
            # In some regions, 4 is 'tư' in certain positions,
            # but usually TTS uses 'bốn' unless specified.
            # The previous implementation used 'bốn'.
            res.append(units[u_digit])
        else:
            res.append(units[u_digit])

    return " ".join(res)

def n2w_large_number(numbers: str) -> str:
    """Convert large numbers to Vietnamese words."""
    if not numbers or not numbers.lstrip('0'):
        return units['0']
        
    # Remove leading zeros
    numbers = numbers.lstrip('0')

    # Split into 3-digit groups from right to left
    n_len = len(numbers)
    groups = []
    for i in range(n_len, 0, -3):
        groups.append(numbers[max(0, i-3):i])

    suffixes = ['', ' nghìn', ' triệu', ' tỷ']

    parts = []
    for i, group in enumerate(groups):
        if group == '000':
            continue

        word = n2w_hundreds(group)
        if word:
            # Suffix handling
            # 0: none, 1: nghìn, 2: triệu, 3: tỷ, 4: nghìn (tỷ), 5: triệu (tỷ), 6: tỷ (tỷ)...
            suffix_idx = i % 3
            main_suffix = suffixes[suffix_idx] if suffix_idx < len(suffixes) else ""

            # Billion (tỷ) is special in Vietnamese, it repeats or scales
            tỷ_count = i // 3

            if main_suffix or tỷ_count > 0:
                full_suffix = main_suffix + (" tỷ" * tỷ_count)
                word_with_suffix = f"{word}{full_suffix}"
            else:
                word_with_suffix = word
            parts.append(word_with_suffix)

    if not parts:
        return units['0']

    # Reverse parts and join
    parts.reverse()
    return ' '.join(parts).strip()

def n2w(number: str) -> str:
    """Main entry point for number to word conversion."""
    clean_number = pre_process_n2w(number)
    if not clean_number:
        return str(number)

    if len(clean_number) == 2 and clean_number[0] == '0':
        return f"không {units[clean_number[1]]}"

    return n2w_large_number(clean_number)

def n2w_single(number: str) -> str:
    """Convert number to word by digits (e.g. for phone numbers)."""
    if str(number).startswith('+84'):
        number = '0' + str(number)[3:]
    clean_number = pre_process_n2w(number)
    if not clean_number:
        return str(number)
    return process_n2w_single(clean_number)
