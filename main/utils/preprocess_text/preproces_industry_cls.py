import re

import visen
from unicodedata import normalize
from pyvi import ViTokenizer

### brackets list
opening_ls = ['[', '{', '⁅', '〈', '⎡', '⎢', '⎣', '⎧', '⎨', '⎩', '❬', '❰', '❲', '❴', '⟦', '⟨', '⟪', '⟬', '⦃', '⦇', '⦉',
              '⦋', '⦍', '⦏', '⦑', '⦓', '⦕', '⦗', '⧼', '⸂', '⸄', '⸉', '⸌', '⸜', '⸢', '⸤', '⸦', '〈', '《', '「', '『',
              '【', '〔', '〖', '〘', '〚', '﹛', '﹝', '［', '｛', '｢', '｣']

closing_ls = [']', '}', '⁆', '〉', '⎤', '⎥', '⎦', '⎫', '⎬', '⎭', '❭', '❱', '❳', '❵', '⟧', '⟩', '⟫', '⟭', '⦄', '⦈', '⦊',
              '⦌', '⦎', '⦐', '⦒', '⦔', '⦖', '⦘', '⧽', '⸃', '⸅', '⸊', '⸍', '⸝', '⸣', '⸥', '⸧', '〉', '》', '」', '』',
              '】', '〕', '〗', '〙', '〛', '﹜', '﹞', '］', '｝', '｣']

opening_brackets = {key: '(' for key in opening_ls}
closing_brackets = {key: ')' for key in closing_ls}

### constant
PUNC = '!\"#$&()*+,-–−./:;=?@[\]^_`{|}~”“`°²ˈ‐ㄧ‛∼’'  # remove <> for number_sym and unknown_sym
UNKNOWN_SYM = ''  # mix number with character, ex: 6.23A

re_email = '([a-zA-Z0-9._%+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z]{2,4}(\\.?[a-zA-Z]{2,4})?)'
re_url = '(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
re_url2 = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
re_image = '(tập[\s_]tin|hình|file|image|imagesize).*?(jpg|svg|png|gif|jpeg|ogg|tif|width)'
re_num_and_decimal = '[0-9]*[,.\-]*[0-9]*[,.\-]*[0-9]*[.,\-]*[0-9]*[,.\-]*[0-9]+[.,]?'
re_unknown = '[a-z]+[\d]+[\w]*|[\d]+[a-z]+[\w]*'
re_vnese_txt = r'[^a-zA-ZàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ\s]'
re_truncate_unknown = '(<UNKNUM>\s*)+'


special_punc = {'”': '"', '': '', "’": "'", "`": "'"}


def replace_all(replacer: dict, txt: str) -> str:
    for old, new in replacer.items():
        txt = txt.replace(old, new)
    return txt


def replace_num(text: str) -> str:
    text = re.sub(re_num_and_decimal, UNKNOWN_SYM, text)
    return text


def replace_unknown(text: str) -> str:
    text = re.sub(re_unknown, UNKNOWN_SYM, text)
    return text


def unicode_normalizer(text, forms: list = ['NFKC', 'NKFD', 'NFC', 'NFD']) -> str:
    for form in forms:
        text = normalize(form, text)
    return text


def normalize_bracket(text: str) -> str:
    text = replace_all(opening_brackets, text)
    text = replace_all(closing_brackets, text)
    text = re.sub(r"[\(\[].*?[\)\]]", " ", text)
    return text


def remove_punc(text: str) -> str:
    r = re.compile(r'[\s{}]+'.format(re.escape(PUNC)))
    text = r.split(text)
    return ' '.join(i for i in text if i)


def truncate_unknown(text: str) -> str:
    text = re.sub(re_truncate_unknown, UNKNOWN_SYM, text)
    return text


with open('data/stop_words.txt', 'r') as fi:
    stop_words = fi.readlines() 
    stop_words = [word.rstrip() for word in stop_words]


def clean_text(text: str, stop_words=stop_words) -> str:
    text = str(text)
    text = text.split('\n')[0]
    text = unicode_normalizer(text, ["NFKC"])
    text = text.lower()
    text = remove_punc(text)
    text = truncate_unknown(text)
    text = re.sub(re_vnese_txt, " ", text)
    text = text.strip()
    text = ViTokenizer.tokenize(text)
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return visen.clean_tone(text)


if __name__ == '__main__':
    txt = ['{DATE 5/2023} SỮA NAN NGA 1 1.800G (6LON/THÙNG)*ĐI ĐƠN KHI ĐỦ THÙNG* (HỘP) 😂❤😍😒😢😉😜🎉',
           'S{PFGdjgjfghfjjh',
           '{Tem phụ} Mặt nạ giấy dưỡng da cao cấp Whisis Hàn Quốc 25ml',
           'Nước tẩy trang innisfree trà xanh mẫu mới ( Không Bill) [ Date 30/6/2024',
           '[Sữa Tắm Sáng Da Reihaku Hatomugi 800ml',
           '[K.V[ DHA Bio Island Cho Bà Bầu 60 Viên( Mẫu mới) 123 456 789 asa4asfa535 gekko',
           'Mặt Nạ Đất Sét Bạc Hà Dreamworks I\'m The Real Shrek Pack 110g - 123 456 789 asa4asfa535',
           'NORMADERM PHYTOSOLUTION DOUBLE- CORRECTION DAILY CARE \"KEM DƯỠNG DẠNG GEL SỮA DÀNH CHO DA MỤN VỚI TÁC ĐỘNG KÉP GIÚP GIẢM MỤN, GIẢM KHUYẾT ĐIỂM TRÊN DA & GIÚP PHỤC HỒI, DƯỠNG ẨM LÀN DA\"',
           '24Hr CREAM DEODORANT SENSITIVE OR DEPILATED SKIN \"KEM KHỬ MÙI VÀ GIÚP DƯỠNG DA MỀM MỊN (CHO LÀN DA NHẠY CẢM)\"',
           "Tẩy tế bào chết cho da mặt \"Mật ong và Cafe\", Anteka 75ml anteka",
           'QUẦN JEAN LOUIS VUITTON T20T128',
           '𝓗𝓾𝔂̀𝓷𝓱 𝓛𝓪̂𝓶 𝓣𝓸̂́ 𝓝𝓰𝓪̂𝓷',
           'sữa i\'m ensure']

    for i in txt:
        print(clean_text(i))



    
