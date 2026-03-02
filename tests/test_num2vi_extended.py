import pytest
from vieneu_utils.cleaner.num2vi import n2w, n2w_single

def test_n2w_zero_padding():
    assert n2w("007") == "bảy"
    assert n2w("010") == "mười"
    assert n2w("000") == "không"

def test_n2w_large_scales():
    # 1,000,000,000,000 (1 trillion) - In VN usually "một nghìn tỷ"
    assert n2w("1000000000000") == "một nghìn tỷ"
    # 1,000,000,000,000,000 (1 quadrillion) - "một triệu tỷ"
    assert n2w("1000000000000000") == "một triệu tỷ"
    # 1,000,000,000,000,000,000 (1 quintillion) - "một tỷ tỷ"
    assert n2w("1000000000000000000") == "một tỷ tỷ"

def test_n2w_mot_vs_mot():
    assert n2w("21") == "hai mươi mốt"
    assert n2w("11") == "mười một"
    assert n2w("1") == "một"
    assert n2w("101") == "một trăm lẻ một"
    assert n2w("121") == "một trăm hai mươi mốt"

def test_n2w_lam_vs_nam():
    assert n2w("5") == "năm"
    assert n2w("15") == "mười lăm"
    assert n2w("25") == "hai mươi lăm"
    assert n2w("105") == "một trăm lẻ lăm"

def test_n2w_single_extended():
    assert n2w_single("0912") == "không chín một hai"
    assert n2w_single("+84912") == "không chín một hai"

def test_n2w_hundreds_internal():
    from vieneu_utils.cleaner.num2vi import n2w_hundreds
    assert n2w_hundreds("005") == "không trăm lẻ lăm"
    assert n2w_hundreds("015") == "không trăm mười lăm"
    assert n2w_hundreds("100") == "một trăm"
