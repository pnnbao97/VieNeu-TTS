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

def chunks(lst, n):
    chunks_lst = []
    for i in range(0, len(lst), n):
        chunks_lst.append(lst[i : i + n])
    return chunks_lst

def n2w_units(numbers: str):
    if not numbers:
        raise ValueError('Số rỗng!!')
    if len(numbers) > 1:
        raise ValueError('Số vượt quá giá trị của hàng đơn vị!')
    return units[numbers]

def pre_process_n2w(number: str):
    char_to_replace = {' ': '', '-': '', '.': '', ',': ''}
    for key, value in char_to_replace.items():
        number = str(number).replace(key, value)
    if not number.isdigit():
        return None
    return number

def process_n2w_single(numbers: str):
    total_number = ''
    for element in numbers:
        if element in units:
            total_number += units[element] + ' '
    return total_number.strip()

def n2w_hundreds(numbers: str):
    if len(numbers) > 3 or len(numbers) == 0:
        return ""
    if len(numbers) <= 1:
        return n2w_units(numbers)
    
    reversed_hundreds = numbers[::-1]
    total_number = []
    for e in range(0, len(reversed_hundreds)):
        if e == 0:
            total_number.append(units[reversed_hundreds[e]])
        elif e == 1:
            total_number.append(units[reversed_hundreds[e]] + ' mươi ')
        elif e == 2:
            total_number.append(units[reversed_hundreds[e]] + ' trăm ')

    for idx, value in enumerate(total_number):
        if idx == 0 and value == 'không':
            if len(total_number) > 1 and total_number[1] == 'không mươi ':
                total_number[1] = ''
            total_number[idx] = ''
        if idx == 0 and value == 'một':
            if len(total_number) > 1 and total_number[1] != 'một mươi ' and total_number[1] != 'không mươi ':
                total_number[idx] = 'mốt'
        if idx == 0 and value == 'năm':
            if len(total_number) > 1 and total_number[1] != 'không mươi ':
                total_number[idx] = 'lăm'
        if value == 'không mươi ':
            total_number[idx] = 'lẻ '
        if value == 'một mươi ':
            total_number[idx] = 'mười '

    return ''.join(total_number[::-1]).strip()

def n2w_large_number(numbers: str):
    if int(numbers) == 0:
        return units['0']
        
    reversed_large_number = numbers[::-1]
    reversed_large_number = chunks(reversed_large_number, 3)

    suffixes = ['', ' nghìn ', ' triệu ', ' tỷ ']
    parts = []
    for e in range(len(reversed_large_number)):
        chunk = reversed_large_number[e][::-1]
        if int(chunk) == 0:
            continue
        word = n2w_hundreds(chunk)
        if e < len(suffixes):
            word += suffixes[e]
        parts.append(word)

    return ' '.join(p.strip() for p in parts[::-1]).strip()

def n2w(number: str):
    clean_number = pre_process_n2w(number)
    if not clean_number: return str(number)
    return n2w_large_number(clean_number)

def n2w_single(number: str):
    if str(number).startswith('+84'):
        number = str(number).replace('+84', '0')
    clean_number = pre_process_n2w(number)
    if not clean_number: return str(number)
    return process_n2w_single(clean_number)
