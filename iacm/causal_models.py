from typing import Dict, Any, List
import numpy as np
from iacm.utils import insert, flatten, count_char

obj_fct_weights_for_binary_case = [{0: 3, 1: 3, 13: 0, 15: 0},
                                   {13: 3, 15: 3, 0: 0, 1: 0},
                                   {6: 3, 7: 3, 8: 0, 10: 0},
                                   {8: 3, 10: 3, 6: 0, 7: 0}]

causal_model_definition = dict()
causal_model_definition['X|Y'] = {'V': 'X,Y,X_y,Y_x', 'nb_observed_variables': 2,
                                  'interventional_variables': ['X', 'Y'], 'hidden_variables': [], 'obj_fct_weights': []}
causal_model_definition['X->Y'] = {'V': 'X,Y,Y_x', 'nb_observed_variables': 2, 'interventional_variables': ['X'],
                                   'hidden_variables': [], 'obj_fct_weights': []}
causal_model_definition['X->Y+'] = {'V': 'X,Y,Y_x', 'nb_observed_variables': 2, 'interventional_variables': ['X'],
                                    'hidden_variables': [], 'obj_fct_weights': obj_fct_weights_for_binary_case}
causal_model_definition['X<-Z->Y'] = {'V': 'X,Y,Z,X_z,Y_z', 'nb_observed_variables': 3,
                                      'interventional_variables': ['Z'], 'hidden_variables': [], 'obj_fct_weights': []}
causal_model_definition['X<-[Z]->Y'] = {'V': 'X,Y,X_z,Y_z', 'nb_observed_variables': 2, 'interventional_variables': [],
                                        'hidden_variables': ['Z'], 'obj_fct_weights': []}
causal_model_definition['Z->X->Y'] = {'V': 'X,Y,Z,X_z,Y_x', 'nb_observed_variables': 3,
                                      'interventional_variables': ['Z', 'X'], 'hidden_variables': [],
                                      'obj_fct_weights': []}
causal_model_definition['[Z]->X->Y'] = {'V': 'X,Y,X_z,Y_x', 'nb_observed_variables': 2, 'interventional_variables': [],
                                        'hidden_variables': ['Z'], 'obj_fct_weights': []}
causal_model_definition['(X,Z)->Y'] = {'V': 'X,Y,Z,Y_x,Y_z,', 'nb_observed_variables': 3,
                                       'interventional_variables': ['X', 'Z'], 'hidden_variables': [],
                                       'obj_fct_weights': []}
causal_model_definition['(X,[Z])->Y'] = {'V': 'X,Y,Z,Y_x,Y_z,', 'nb_observed_variables': 2,
                                         'interventional_variables': ['X'], 'hidden_variables': ['Z'],
                                         'obj_fct_weights': []}


def setup_causal_model_data(base: int, causal_model: Dict[str, Any], monotone_incr=False, monotone_decr=False) \
        -> Dict[str, Any]:
    observation_variables = causal_model['V'].split(',')[0:causal_model['nb_observed_variables']]
    observed_variables_after_intervention = causal_model['V'].split(',')[causal_model['nb_observed_variables']:]
    nb_observed_variables = len(observation_variables)
    nb_observed_variables_after_intervention = len(observed_variables_after_intervention)
    nb_variables = nb_observed_variables + nb_observed_variables_after_intervention * base
    size_prob = pow(base, nb_variables)
    causal_model_data = dict()
    causal_model_data['base_x'] = base
    causal_model_data['base_y'] = base
    causal_model_data['nb_variables'] = nb_variables
    causal_model_data['size_prob'] = size_prob
    causal_model_data['interventional_variables'] = causal_model['interventional_variables']
    causal_model_data['hidden_variables'] = causal_model['hidden_variables']
    causal_model_data['V'] = causal_model['V']
    causal_model_data['obj_fct_weights'] = causal_model['obj_fct_weights']

    pattern_data = generate_pattern_data(base=base, observation_variables=observation_variables,
                                         observed_variables_after_intervention=observed_variables_after_intervention,
                                         interventional_variables=causal_model['interventional_variables'],
                                         hidden_variables=causal_model['hidden_variables'])
    causal_model_data['constraint_patterns'] = pattern_data[base]['constraint_patterns']

    lines = generate_constraint_lines(pattern_data[base]['constraint_patterns'], size_prob, base)
    causal_model_data['B'] = np.array([[1] * size_prob] + lines)

    causal_model_data['F'] = np.diag(np.array([1] * size_prob))
    causal_model_data['c'] = np.array([0.0] * size_prob)

    zero_codes = get_zero_codes(pattern_data[base]['zero_code_patterns'], base)
    if monotone_decr:
        zero_codes.append('0001')
        zero_codes.append('1101')
    if monotone_incr:
        zero_codes.append('0110')
        zero_codes.append('1010')
    causal_model_data['S_codes'] = s_codes(zero_codes, base, nb_variables)
    d_list = list()
    for i in range(0, size_prob):
        if base_repr(i, base, nb_variables) in causal_model_data['S_codes']:
            d_list.append(1)
            causal_model_data['c'][i] = 0.00001
        else:
            d_list.append(0)
    causal_model_data['d'] = np.array(d_list)

    return causal_model_data


def generate_pattern_data(base: int, observation_variables: List[str], observed_variables_after_intervention: List[str],
                          interventional_variables: List[str], hidden_variables: List[str]) -> Dict:
    result_dict = dict()
    result_dict[base] = dict()
    nb_observations = len(observation_variables)
    nb_observed_variables_after_intervention = pow(base, len(observed_variables_after_intervention))
    pattern_template = 'x'*nb_observations + 'x'*nb_observed_variables_after_intervention
    intervention_template = 'x' * nb_observed_variables_after_intervention
    total_len = nb_observations + nb_observed_variables_after_intervention

    # generate constraint_patterns
    constraint_patterns = list()
    for intervention in range(1, base):
        for position in range(0, nb_observed_variables_after_intervention):
            constraint_patterns.append(insert(pattern_template, str(intervention), total_len - position))

    # iterate through the ranges of observed variables
    for observed_value in range(pow(base, nb_observations)-1, -1, -1):
        if observed_value == 0:
            continue
        constraint_patterns.append(insert(pattern_template, base_repr(observed_value, base, nb_observations),
                                          nb_observations))

    # generate zero_code_patterns
    zero_code_patterns = list()

    observation_variables_mapping = {k: v for v, k in enumerate(observation_variables)}
    observed_variables_after_intervention_mapping = {k: v for v, k in enumerate(observed_variables_after_intervention)}
    hidden_variables_mapping = {k: v for v, k in enumerate(hidden_variables)}

    all_interventional_variables = interventional_variables + hidden_variables

    for observed_value in range(0, pow(base, nb_observations)):
        for intervention_value in range(0, pow(base, nb_observed_variables_after_intervention)):
            observation_data = base_repr(observed_value, base, nb_observations)
            intervention_data = base_repr(intervention_value, base, nb_observed_variables_after_intervention)

            for observation_var, observation_var_index in observation_variables_mapping.items():
                for intervention_var in all_interventional_variables:

                    if observation_var == intervention_var:
                        continue
                    observation_var_after_intervention = observation_var + '_' + intervention_var.lower()
                    if observation_var_after_intervention in observed_variables_after_intervention_mapping.keys():
                        adjusted_variable_index = \
                            observed_variables_after_intervention_mapping[observation_var_after_intervention]
                    else:
                        continue
                    if intervention_var in observation_variables_mapping:
                        intervention_index = observation_variables_mapping[intervention_var]
                        observation_data_at_intervention_index = int(observation_data[intervention_index])
                        if observation_data[observation_var_index] != intervention_data[adjusted_variable_index * base + observation_data_at_intervention_index]:
                            intervention_pattern = insert(intervention_template, intervention_data[
                                adjusted_variable_index * base + observation_data_at_intervention_index],
                                                          adjusted_variable_index * base + observation_data_at_intervention_index + 1)
                            zero_code_patterns.append(observation_data + intervention_pattern)
                    else:
                        for hidden_variable_values in range(0, pow(base, len(hidden_variables))):
                            hidden_data = base_repr(hidden_variable_values, base, len(hidden_variables))
                            for hidden_variable in hidden_variables:
                                observation_data_at_intervention_index = int(hidden_data[hidden_variables_mapping[hidden_variable]])
                                if observation_data[observation_var_index] != intervention_data[adjusted_variable_index*base + observation_data_at_intervention_index]:
                                    intervention_pattern = insert(intervention_template, intervention_data[adjusted_variable_index*base + observation_data_at_intervention_index],
                                                                  adjusted_variable_index*base + observation_data_at_intervention_index + 1)
                                    zero_code_patterns.append(observation_data + intervention_pattern)

    zero_code_patterns = list(set(zero_code_patterns))

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
    return sum([p[replace_char_by_char('x', fixed_code, format_str.format(code_nb))]
                for code_nb in range(0, pow(2, nb_x))])
