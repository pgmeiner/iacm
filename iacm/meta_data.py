from typing import Dict
import numpy as np
from utils import insert, flatten, count_char


def setup_meta_data(base, nb_variables, monotone_incr=False, monotone_decr=False):
    meta_data = dict()
    size_prob = pow(base, nb_variables)
    meta_data['base_x'] = base
    meta_data['nb_variables'] = nb_variables
    meta_data['size_prob'] = size_prob

    pattern_data = generate_pattern_data(base=base, nb_interventions=nb_variables-2)
    meta_data['constraint_patterns'] = pattern_data[base]['constraint_patterns']

    lines = generate_constraint_lines(pattern_data[base]['constraint_patterns'], size_prob, base)
    meta_data['B'] = np.array([[1] * size_prob] + lines)

    zero_codes = get_zero_codes(pattern_data[base]['zero_code_patterns'], base)
    if monotone_decr:
        zero_codes.append('0001')
        zero_codes.append('1101')
    if monotone_incr:
        zero_codes.append('0110')
        zero_codes.append('1010')
    meta_data['S_codes'] = s_codes(zero_codes, base, nb_variables)
    d_list = list()
    for i in range(0, size_prob):
        if base_repr(i, base, nb_variables) in meta_data['S_codes']:
            d_list.append(1)
        else:
            d_list.append(0)
    meta_data['d'] = np.array(d_list)

    meta_data['F'] = np.diag(np.array([1] * size_prob))
    meta_data['c'] = np.array([0.0] * size_prob)

    return meta_data


# pattern_data = {2: {'constraint_patterns': ['xxx1', 'xx1x', '11xx', '10xx', '01xx'],
#                     'zero_code_patterns': ['001x', '010x', '10x1', '11x0']},# '1001', '0101']},
#                 3: {'constraint_patterns': ['xxxx1', 'xxx1x', 'xx1xx', 'xxxx2', 'xxx2x', 'xx2xx',
#                                             '22xxx', '21xxx', '20xxx', '12xxx', '11xxx', '10xxx', '02xxx', '01xxx'],
#                     'zero_code_patterns': ['001xx', '002xx', '010xx', '012xx', '020xx', '021xx',
#                                            '10x1x', '10x2x', '11x0x', '11x2x', '12x0x', '12x1x',
#                                            '20xx1', '20xx2', '21xx0', '21xx2', '22xx0', '22xx1']}
#                 }

def generate_pattern_data(base: int, nb_interventions: int) -> Dict:
    result_dict = dict()
    result_dict[base] = dict()
    pattern_template = 'x'*2 + 'x'*nb_interventions
    total_len = 2 + nb_interventions

    # generate constraint_patterns
    constraint_patterns = list()
    for intervention in range(1, base):
        for position in range(0, nb_interventions):
            constraint_patterns.append(insert(pattern_template, str(intervention), total_len - position))

    # iterate through the ranges of X and Y
    for value_x in range(base-1, -1, -1):
        for value_y in range(base-1, -1, -1):
            if (value_x == 0) and (value_y == 0):
                continue
            constraint_patterns.append(insert(pattern_template, str(value_x) + str(value_y), 2))

    # generate zero_code_patterns
    zero_code_patterns = list()

    for value_x in range(0, base):
        for value_y in range(0, base,):
            for intervention_y in range(0, base):
                if value_y != intervention_y:
                    zero_code_patterns.append(
                        insert(insert(pattern_template, str(value_x) + str(value_y), 2), str(intervention_y), 3 + value_x))

    result_dict[base]['constraint_patterns'] = constraint_patterns
    result_dict[base]['zero_code_patterns'] = zero_code_patterns

    return result_dict


def generate_constraint_lines(patterns, size_prob, base):
    lines = list()
    for pattern in patterns:
        lines.append(convert_to_constraint_line(generate_codes(pattern, base), size_prob, base))

    return lines


def s_codes(zero_codes, base, nb_variables):
    all_codes = generate_codes('x'*nb_variables, base)
    return list(set(all_codes) - set(zero_codes))


def get_zero_codes(code_patterns, base):
    codes = flatten([generate_codes(code_pattern, base) for code_pattern in code_patterns])
    return codes


def convert_to_constraint_line(codes, size_prob, base):
    positions = list()
    for code in codes:
        positions.append(int(code, base))
    result = list()
    for i in range(0, size_prob):
        if i in positions:
            result.append(1)
        else:
            result.append(0)
    return result


def generate_codes(pattern, base):
    nb_x = count_char(pattern, 'x')
    codes = [replace_char_by_char('x', pattern, base_repr(nb, base, nb_x)) for nb in range(0, pow(base, nb_x))]
    return codes


def base_repr(number, base, str_len):
    repr = np.base_repr(number, base)
    return '0'*(str_len-len(repr)) + repr


def replace_char_by_char(char_to_replaced, str_to_be_replaced, to_be_inserted_str):
    replace_index = 0
    final_str = ''
    for i, c in enumerate(str_to_be_replaced):
        if c == char_to_replaced:
            final_str = final_str + to_be_inserted_str[replace_index]
            replace_index = replace_index + 1
        else:
            final_str = final_str + c

    return final_str


def marginal_distribution(p, fixed_code):
    nb_x = count_char(fixed_code, 'x')
    format_str = '{0:0xb}'.replace('x', str(nb_x))
    return sum([p[replace_char_by_char('x', fixed_code, format_str.format(code_nb))] for code_nb in range(0, pow(2, nb_x))])
