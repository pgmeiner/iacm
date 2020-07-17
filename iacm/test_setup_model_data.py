import numpy as np
from iacm.causal_models import setup_model_data, setup_causal_model_data, causal_model_definition

expected_model_data = dict()
expected_model_data['2_2'] = {
    'base_x': 2, 'nb_variables': 4, 'size_prob': 16,
    'constraint_patterns': ['xxx1', 'xx1x', '11xx', '10xx', '01xx'],
    'B': np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]),
    'S_codes': ['0111', '1101', '1000', '0110', '1010', '1111', '0001', '0000'],
    'd': np.array([1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1]),
    'F': np.diag(np.array([1] * pow(2, 4))),
    'c': np.array([0.0] * pow(2, 4))}

expected_model_data['2_2_m_d'] = {
    'base_x': 2, 'nb_variables': 4, 'size_prob': 16,
    'constraint_patterns': ['xxx1', 'xx1x', '11xx', '10xx', '01xx'],
    'B': expected_model_data['2_2']['B'],
    'S_codes': ['0110', '1111', '1010', '0000', '1000', '0111'],
    'd': np.array([1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1]),
    'F': np.diag(np.array([1] * pow(2, 4))),
    'c': np.array([0.0] * pow(2, 4))}

expected_model_data['2_2_m_i'] = {
    'base_x': 2, 'nb_variables': 4, 'size_prob': 16,
    'constraint_patterns': ['xxx1', 'xx1x', '11xx', '10xx', '01xx'],
    'B': expected_model_data['2_2']['B'],
    'S_codes': ['0001', '1111', '0000', '1000', '1101', '0111'],
    'd': np.array([1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1]),
    'F': np.diag(np.array([1] * pow(2, 4))),
    'c': np.array([0.0] * pow(2, 4))}

expected_model_data['3_3'] = {
    'base_x': 3, 'nb_variables': 5, 'size_prob': 243,
    'constraint_patterns': ['xxxx1', 'xxx1x', 'xx1xx', 'xxxx2', 'xxx2x', 'xx2xx', '22xxx', '21xxx', '20xxx', '12xxx', '11xxx', '10xxx', '02xxx', '01xxx'],
    'S_codes': ['20110', '21201', '00001', '20210', '00002', '20000', '20010', '02222', '01112', '12120', '21121', '10001', '22212', '12022', '01101', '01122', '00000', '21111', '21221', '00011', '21001', '20100', '01102', '10002', '12220', '22012', '01110', '22002', '20120', '12121', '22022', '12221', '11011', '22122', '20020', '12021', '02201', '01120', '21021', '21101', '10201', '11110', '12020', '11112', '22102', '02200', '10200', '01111', '10000', '10100', '10101', '11111', '21011', '00020', '02202', '02211', '20200', '22222', '22112', '11212', '00010', '12222', '01121', '02220', '11211', '22202', '21211', '20220', '01100', '02212', '00021', '00022', '11010', '00012', '02210', '10102', '11012', '12122', '10202', '02221', '11210'],
    'F': np.diag(np.array([1] * pow(3, 5))),
    'c': np.array([0.0] * pow(3, 5))}

expected_model_data['4_4'] = {
    'base_x': 4, 'nb_variables': 6, 'size_prob': 4096,
    'constraint_patterns': ['xxxxx1', 'xxxx1x', 'xxx1xx', 'xx1xxx', 'xxxxx2', 'xxxx2x', 'xxx2xx', 'xx2xxx', 'xxxxx3', 'xxxx3x', 'xxx3xx', 'xx3xxx', '33xxxx', '32xxxx', '31xxxx', '30xxxx', '23xxxx', '22xxxx', '21xxxx', '20xxxx', '13xxxx', '12xxxx', '11xxxx', '10xxxx', '03xxxx', '02xxxx', '01xxxx'],
    'S_codes': ['122232', '011121', '132312', '330113', '323202', '011032', '302210', '033202', '300030', '312001', '321212', '122211', '110130', '201003', '033002', '011101', '223223', '301330', '231331', '102021', '312201', '011001', '011111', '303210', '000101', '232330', '022303', '000232', '233031', '131331', '333213', '332023', '223120', '033023', '332033', '033033', '011233', '231133', '333223', '311231', '330023', '231230', '211211', '100002', '000302', '011332', '113113', '131332', '320232', '033010', '112100', '210010', '303200', '111121', '332213', '011133', '033030', '133331', '111100', '112113', '222022', '022033', '133320', '202300', '000213', '300130', '300300', '230332', '333113', '233330', '121223', '102013', '322122', '011012', '203100', '102002', '323332', '332313', '210213', '233132', '131323', '323232', '210312', '333123', '102000', '131302', '130321', '033213', '300210', '220021', '022203', '301220', '321002', '222020', '323302', '311001', '130301', '211111', '210311', '311221', '213313', '323132', '311331', '313121', '231131', '000300', '232233', '000201', '112121', '000222', '203102', '331133', '102031', '223122', '022323', '322022', '222021', '102020', '300320', '331203', '232130', '301020', '203203', '100003', '000033', '223220', '100022', '102012', '312111', '322312', '301110', '200200', '011033', '330233', '333033', '112130', '110121', '033130', '121232', '022220', '313101', '100030', '000032', '320102', '022120', '102030', '022231', '133300', '123202', '103022', '101012', '331323', '223222', '130330', '022331', '011022', '033212', '113112', '231132', '301010', '122202', '332113', '310001', '331033', '323112', '122210', '313001', '011002', '303010', '121212', '313131', '320322', '220322', '113132', '022210', '202100', '321202', '022300', '333333', '221320', '212212', '303130', '033320', '321132', '122203', '202303', '022020', '311011', '332333', '220123', '210113', '022131', '000210', '310011', '322232', '323312', '310311', '230030', '200102', '213210', '123212', '113122', '210212', '313011', '120201', '000233', '000212', '213110', '011312', '202001', '000011', '332303', '000031', '132323', '311121', '122200', '231231', '011313', '121202', '230333', '233033', '233130', '310211', '101031', '211110', '303110', '232033', '321332', '120211', '033330', '302030', '022310', '033120', '211210', '011231', '000320', '333303', '212113', '033313', '033102', '111101', '022123', '331023', '222122', '033000', '302230', '110110', '330123', '231332', '011003', '000301', '231130', '121211', '233233', '321232', '100013', '221321', '102023', '303120', '011310', '321312', '311301', '011030', '103000', '202301', '000123', '111123', '313201', '022133', '022030', '122213', '331313', '220020', '000221', '323012', '221023', '120231', '311031', '133303', '302010', '330323', '221020', '231330', '000200', '103002', '233030', '331013', '011211', '120213', '320222', '022211', '022132', '123233', '113123', '122212', '120202', '233232', '033003', '220223', '000022', '022313', '223123', '120233', '320132', '033122', '000013', '110133', '111110', '011123', '230330', '133312', '033322', '120223', '113133', '011112', '333233', '133321', '223320', '322132', '101000', '130320', '120203', '232030', '123230', '011220', '033231', '302330', '000220', '322202', '332013', '301320', '111130', '011322', '320122', '312301', '312221', '230033', '033211', '033300', '033032', '331113', '011320', '101011', '332003', '123231', '221323', '232232', '000020', '211011', '000331', '112133', '320332', '000313', '101010', '233333', '130323', '200001', '302300', '123203', '210111', '313111', '033222', '000312', '112101', '200101', '331233', '033001', '220221', '303100', '022230', '201202', '131322', '202000', '301200', '101021', '301300', '330223', '110122', '120210', '200201', '011311', '011013', '221322', '333013', '022222', '011023', '000102', '321112', '302110', '310231', '122220', '221220', '223321', '022223', '000030', '200002', '022320', '332103', '312131', '320022', '102003', '213111', '011120', '300310', '322032', '332323', '302130', '033110', '123201', '022233', '123220', '200300', '113131', '111120', '330213', '221120', '302220', '011222', '011031', '133310', '033332', '132321', '100033', '310321', '101033', '222223', '333203', '212313', '201200', '121201', '213010', '022221', '011010', '320032', '133301', '022232', '212311', '320202', '123210', '103032', '033233', '331213', '120222', '011102', '100020', '033311', '033321', '222222', '011131', '022013', '213213', '323322', '313231', '300330', '321032', '113110', '000001', '022110', '011113', '310221', '323022', '201002', '310131', '203300', '123223', '333103', '123211', '130322', '000131', '233231', '000311', '220122', '120212', '320112', '113111', '120221', '302020', '233331', '210310', '033132', '132332', '131312', '112112', '330133', '120230', '011020', '311111', '022301', '132322', '202200', '313311', '022101', '300230', '133333', '232333', '213012', '312031', '033111', '130311', '332233', '100000', '232331', '223121', '231030', '230130', '033203', '222120', '203002', '320312', '203303', '212312', '132302', '220220', '200003', '011103', '022032', '201300', '100031', '000110', '323002', '022111', '000202', '223323', '233032', '112131', '201203', '011303', '131300', '213312', '133330', '200303', '221223', '121231', '111133', '110112', '202302', '113100', '022100', '022022', '000012', '033013', '022021', '322302', '122233', '333003', '011100', '310121', '110111', '220023', '000103', '203201', '000121', '101003', '300000', '212211', '211212', '000000', '132333', '130331', '000332', '022200', '230131', '222323', '310021', '022311', '103001', '230132', '101001', '310101', '022202', '022212', '000122', '103020', '011333', '301000', '223221', '212013', '202002', '230231', '112120', '011302', '312331', '233131', '202003', '200103', '033133', '100010', '033331', '333133', '311201', '201201', '103031', '300020', '123213', '110102', '330313', '132310', '200202', '033230', '111102', '331003', '011201', '033323', '113101', '222321', '301030', '301230', '210211', '213011', '201000', '320002', '132320', '301210', '331223', '011230', '222220', '231333', '211112', '231032', '210313', '202201', '302310', '320302', '022001', '221221', '201102', '203202', '011331', '301130', '033201', '232133', '102011', '333323', '332123', '110123', '101032', '113120', '200203', '122221', '033200', '230233', '222121', '202103', '212213', '132331', '123221', '313021', '022322', '223023', '212012', '321012', '103013', '330003', '233133', '113121', '103030', '232332', '211012', '131333', '200000', '130310', '132330', '120220', '222221', '121203', '132313', '332203', '033302', '000333', '011200', '112102', '322102', '033312', '022321', '101020', '323122', '330013', '000021', '112110', '110113', '033221', '121221', '310031', '033020', '011232', '121220', '323212', '323102', '212112', '313221', '221121', '033131', '303320', '101030', '330033', '102033', '303030', '011210', '102001', '121200', '233230', '111112', '000133', '110120', '231232', '121222', '000230', '133332', '212111', '330203', '000223', '311021', '033100', '220323', '213013', '300200', '213113', '300010', '133322', '132311', '101022', '222023', '022002', '131301', '230331', '311321', '120200', '223021', '113102', '211311', '220121', '033310', '213311', '130300', '113103', '000321', '200301', '101023', '300220', '000130', '033232', '311311', '202203', '200100', '112122', '122201', '000323', '202102', '303310', '000303', '132303', '103003', '223322', '203101', '111122', '321322', '221122', '011202', '101002', '310331', '033223', '033012', '022003', '022011', '313211', '011021', '000322', '220320', '033103', '102032', '120232', '131303', '123200', '011221', '011223', '201303', '110103', '131320', '323032', '213112', '300120', '033123', '221222', '232231', '022333', '211113', '333023', '322112', '320212', '321022', '121233', '033301', '203103', '212210', '033121', '000310', '322212', '312211', '313321', '210012', '033210', '022113', '033101', '211213', '331103', '333313', '000120', '022332', '011122', '300100', '302200', '231031', '203003', '210013', '123222', '033022', '110101', '313301', '000002', '132301', '112103', '222320', '201103', '000231', '213211', '203302', '303330', '312321', '022000', '203301', '311211', '210110', '122231', '133302', '331303', '000113', '022213', '210210', '203001', '222322', '011212', '110131', '130302', '200302', '103023', '220022', '230232', '312011', '022112', '303000', '100032', '100011', '000132', '312021', '230133', '011132', '220222', '130303', '111131', '212110', '232132', '011301', '313031', '322332', '103010', '113130', '230230', '011330', '033113', '321302', '033011', '131310', '011321', '110132', '212010', '033021', '301310', '321122', '122230', '111132', '022102', '303020', '303230', '231033', '132300', '121210', '312101', '022023', '011011', '231233', '232131', '111113', '313331', '230031', '011130', '101013', '103011', '022103', '332133', '130332', '033031', '310201', '133313', '121230', '102022', '221022', '221021', '301100', '131311', '033220', '000203', '213212', '022031', '312121', '220321', '220120', '202202', '330103', '201101', '033333', '323222', '022130', '203000', '322012', '022010', '320012', '011213', '022330', '100001', '133323', '103033', '302000', '331123', '310111', '201001', '222123', '311101', '131313', '011110', '310301', '201302', '223022', '213310', '133311', '232032', '102010', '022201', '331333', '000112', '022012', '211010', '022122', '122223', '233332', '330303', '321102', '111111', '100023', '011000', '303220', '022312', '202101', '011203', '211013', '322322', '100012', '211312', '212011', '223020', '312231', '131321', '000330', '112111', '000010', '112123', '211313', '322002', '103021', '100021', '123232', '332223', '103012', '000211', '033303', '221123', '302320', '232230', '302100', '011300', '022302', '033112', '112132', '301120', '230032', '201100', '210011', '000111', '211310', '212310', '203200', '131330', '130333', '000003', '122222', '322222', '232031', '111103', '210112', '303300', '130313', '022121', '300110', '130312', '321222', '121213', '011323', '302120', '311131', '110100', '000100', '201301', '312311', '330333', '000023'],
    'F': np.diag(np.array([1] * pow(4, 6))),
    'c': np.array([0.0] * pow(4, 6))}

expected_model_data['X|Y'] = {
    'base_x': 2, 'nb_variables': 6, 'size_prob': 64,
    'constraint_patterns': ['xxxxx1', 'xxxx1x', 'xxx1xx', 'xx1xxx', '11xxxx', '10xxxx', '01xxxx'],
    'S_codes': ['100011', '000001', '000101', '100010', '101010', '101011', '011000', '000000', '011010', '011110', '111101', '011100', '000100', '111111', '110111', '110101'],
    'F': np.diag(np.array([1] * pow(2, 6))),
    'c': np.array([0.0] * pow(2, 6))}

expected_model_data['2_2_X<-Z->Y'] = {
    'base_x': 2, 'nb_variables': 7, 'size_prob': 128,
    'constraint_patterns': ['xxxxxx1', 'xxxxx1x', 'xxxx1xx', 'xxx1xxx', '111xxxx', '110xxxx', '101xxxx', '100xxxx', '011xxxx', '010xxxx', '001xxxx'],
    'S_codes': ['0000001', '1110111', '0100010', '1110101', '1001101', '0100110', '1101010', '0100011', '0111001', '1011100', '0110001', '0100111', '1111101', '0000101', '1101110', '1001000', '1001001', '0000000', '1010110', '0011010', '1101111', '1011110', '0010000', '0000100', '0011000', '1111111', '1001100', '0110011', '0010010', '1010100', '0111011', '1101011'],
    'F': np.diag(np.array([1] * pow(2, 7))),
    'c': np.array([0.0] * pow(2, 7))}

expected_model_data['2_2_X<-[Z]->Y'] = {
    'base_x': 2, 'nb_variables': 6, 'size_prob': 64,
    'constraint_patterns': ['xxxxx1', 'xxxx1x', 'xxx1xx', 'xx1xxx', '11xxxx', '10xxxx', '01xxxx'],
    'S_codes': ['000000', '111111', '101100', '010011'],
    'F': np.diag(np.array([1] * pow(2, 6))),
    'c': np.array([0.0] * pow(2, 6))}

expected_model_data['Z->X->Y'] = {
    'base_x': 2, 'nb_variables': 7, 'size_prob': 128,
    'constraint_patterns': ['xxxxxx1', 'xxxxx1x', 'xxxx1xx', 'xxx1xxx', '111xxxx', '110xxxx', '101xxxx', '100xxxx', '011xxxx', '010xxxx', '001xxxx'],
    'S_codes': ['1101111', '1101011', '0111010', '1001000', '0111011', '1011100', '1111101', '0010000', '0011000', '0110011', '1111111', '0100111', '1011110', '0110010', '1001110', '1001100', '0011001', '0100010', '1010100', '1101001', '0000000', '1010110', '0000001', '1101101', '0000100', '1110111', '1110101', '0000101', '0100011', '1001010', '0100110', '0010001'],
    'F': np.diag(np.array([1] * pow(2, 7))),
    'c': np.array([0.0] * pow(2, 7))}

expected_model_data['[Z]->X->Y'] = {
    'base_x': 2, 'nb_variables': 6, 'size_prob': 64,
    'constraint_patterns': ['xxxxx1', 'xxxx1x', 'xxx1xx', 'xx1xxx', '11xxxx', '10xxxx', '01xxxx'],
    'S_codes': ['101100', '010011', '111100', '111111', '000011', '101101', '010010', '101110', '101111', '000001', '000000', '010000', '111110', '111101', '010001', '000010'],
    'F': np.diag(np.array([1] * pow(2, 6))),
    'c': np.array([0.0] * pow(2, 6))}

expected_model_data['(X,Z)->Y'] = {
    'base_x': 2, 'nb_variables': 9, 'size_prob': 512,
    'constraint_patterns': ['xxxxxxxxxx1', 'xxxxxxxxx1x', 'xxxxxxxx1xx', 'xxxxxxx1xxx', 'xxxxxx1xxxx', 'xxxxx1xxxxx', 'xxxx1xxxxxx', 'xxx1xxxxxxx', '111xxxxxxxx', '110xxxxxxxx', '101xxxxxxxx', '100xxxxxxxx', '011xxxxxxxx', '010xxxxxxxx', '001xxxxxxxx'],
    'F': np.diag(np.array([1] * pow(2, 9))),
    'c': np.array([0.0] * pow(2, 9))}

expected_model_data['(X,[Z])->Y'] = {
    'base_x': 2, 'nb_variables': 10, 'size_prob': 1024,
    'constraint_patterns': ['xxxxxxxxxxxxxxxxx1', 'xxxxxxxxxxxxxxxx1x', 'xxxxxxxxxxxxxxx1xx', 'xxxxxxxxxxxxxx1xxx', 'xxxxxxxxxxxxx1xxxx', 'xxxxxxxxxxxx1xxxxx', 'xxxxxxxxxxx1xxxxxx', 'xxxxxxxxxx1xxxxxxx', 'xxxxxxxxx1xxxxxxxx', 'xxxxxxxx1xxxxxxxxx', 'xxxxxxx1xxxxxxxxxx', 'xxxxxx1xxxxxxxxxxx', 'xxxxx1xxxxxxxxxxxx', 'xxxx1xxxxxxxxxxxxx', 'xxx1xxxxxxxxxxxxxx', 'xx1xxxxxxxxxxxxxxx', '11xxxxxxxxxxxxxxxx', '10xxxxxxxxxxxxxxxx', '01xxxxxxxxxxxxxxxx'],
    'F': np.diag(np.array([1] * pow(2, 10))),
    'c': np.array([0.0] * pow(2, 10))}


def test_setup_model_data():
    model_data = dict()
    model_data['X|Y'] = setup_causal_model_data(base=2, causal_model=causal_model_definition['X|Y'])
    model_data['2_2_X<-Z->Y'] = setup_causal_model_data(base=2, causal_model=causal_model_definition['X<-Z->Y'])
    model_data['2_2_X<-[Z]->Y'] = setup_causal_model_data(base=2, causal_model=causal_model_definition['X<-[Z]->Y'])
    model_data['Z->X->Y'] = setup_causal_model_data(base=2, causal_model=causal_model_definition['Z->X->Y'])
    model_data['[Z]->X->Y'] = setup_causal_model_data(base=2, causal_model=causal_model_definition['[Z]->X->Y'])
    model_data['(X,Z)->Y'] = setup_causal_model_data(base=2, causal_model=causal_model_definition['(X,Z)->Y'])
    model_data['(X,[Z])->Y'] = setup_causal_model_data(base=2, causal_model=causal_model_definition['(X,[Z])->Y'])
    model_data['2_2'] = setup_causal_model_data(base=2, causal_model=causal_model_definition['X->Y'])
    model_data['2_2_m_d'] = setup_causal_model_data(base=2, causal_model=causal_model_definition['X->Y'], monotone_decr=True, monotone_incr=False)
    model_data['2_2_m_i'] = setup_causal_model_data(base=2, causal_model=causal_model_definition['X->Y'], monotone_decr=False, monotone_incr=True)
    model_data['3_3'] = setup_causal_model_data(base=3, causal_model=causal_model_definition['X->Y'])
    model_data['4_4'] = setup_causal_model_data(base=4, causal_model=causal_model_definition['X->Y'])

    for signature in ['2_2', '2_2_m_d', '2_2_m_i']:
        m_b_s = model_data[signature]['B'].sort()
        e_b_s = expected_model_data[signature]['B'].sort()
        assert np.array_equal(m_b_s, e_b_s)
        assert_with_sort(model_data[signature]['S_codes'], expected_model_data[signature]['S_codes'])
        assert_with_sort(model_data[signature]['constraint_patterns'], expected_model_data[signature]['constraint_patterns'])
        assert np.array_equal(model_data[signature]['d'], expected_model_data[signature]['d'])
        assert np.array_equal(model_data[signature]['F'], expected_model_data[signature]['F'])
        assert np.array_equal(model_data[signature]['c'], expected_model_data[signature]['c'])

    for signature in ['3_3', '4_4', 'X|Y', '2_2_X<-Z->Y', '2_2_X<-[Z]->Y', 'Z->X->Y', '[Z]->X->Y']:
        assert_with_sort(model_data[signature]['S_codes'], expected_model_data[signature]['S_codes'])
        assert_with_sort(model_data[signature]['constraint_patterns'], expected_model_data[signature]['constraint_patterns'])
        assert np.array_equal(model_data[signature]['F'], expected_model_data[signature]['F'])
        assert np.array_equal(model_data[signature]['c'], expected_model_data[signature]['c'])

    for signature in ['(X,Z)->Y', '(X,[Z])->Y']:
        assert_with_sort(model_data[signature]['constraint_patterns'],expected_model_data[signature]['constraint_patterns'])
        assert np.array_equal(model_data[signature]['F'], expected_model_data[signature]['F'])
        assert np.array_equal(model_data[signature]['c'], expected_model_data[signature]['c'])


def assert_with_sort(p1, p2):
    p1_s = p1.sort()
    p2_s = p2.sort()
    assert p1_s == p2_s
