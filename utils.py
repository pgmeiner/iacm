

def count_char(str, char_to_count):
    count = 0
    for c in str:
        if c == char_to_count:
            count = count + 1
    return count


def insert(source_str: str, insert_str: str, pos: int) -> str:
    return source_str[:pos-len(insert_str)]+insert_str+source_str[pos:]


def flatten(l):
    return [item for sublist in l for item in sublist]