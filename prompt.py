area_info = {
    'qi': 'User Query Inconsistency (QI)',
    'hi': 'Dialogue History Inconsistency (HI)',
    'kbi': 'Knowledge Base Inconsistency (KBI)'
}

detail_info = {
    'qi': 'Query Inconsistency (QI) denotes that the dialogue system response is inconsistent with the current user query. If the assistant\'s reply in the Dialog is consistent with the driver\'s theme in the last turn of Dialog, User Query Inconsistency (QI) is no. Otherwise, yes.',
    'hi': 'Dialogue History Inconsistency (HI) denotes that the dialogue system response is inconsistent with the dialogue history except the current user query. If the assistant does not have multiple rounds (only one assistant turn) of dialogue in the Dialog or the assistant always follows the driver\'s theme, Dialogue History Inconsistency (HI) is no. Otherwise, yes.',
    'kbi': 'Knowledge Base Inconsistency (KBI) denotes that the dialogue system response is inconsistent with the corresponding KB. If the assistant\'s answer in the Dialog does not conflict with the information in the KB, Knowledge Base Inconsistency (KBI) is no. If a conflict or KB lacks information, Knowledge Base Inconsistency (KBI) is yes.'
}

area_detail_baseline_info = f'{detail_info["qi"]} {detail_info["hi"]} {detail_info["kbi"]}'

def get_first_round_prompt(area, dialog, kb):
    prompt = '''You are an expert at determining <area>.

<detail_info>

Please determine whether the following response contains a <area> based on input Dialog<and KB>. Please give your reasons and output the json format of "Answer: {"output": "yes/no"}" so that it can be parsed by the program.

Dialog: <dialog>
KB: <kb>
    '''
    prompt = prompt.replace('<area>', area_info[area])
    prompt = prompt.replace('<detail_info>', detail_info[area])
    # prompt = prompt.replace('<attention>', attention_info[area])

    prompt = prompt.replace('<dialog>', dialog)
    if area == 'kbi':
        prompt = prompt.replace('<and KB>', ' and KB')
        prompt = prompt.replace('<kb>', kb)
    else:
        prompt = prompt.replace('<and KB>', '')
        prompt = prompt.replace('KB: <kb>', '')

    return prompt

# prev_list = {
#     'reason': {
#         'qi': '',
#         'hi': '',
#         'kbi': ''
#     },
#     'pred': {
#         'qi': '',
#         'hi': '',
#         'kbi': ''
#     }
# }
# prediction_id = ['qi', 'hi', 'kbi']
def get_second_round_prompt(area, dialog, kb, prev_list, prediction_id):
#     prompt = '''You are an expert at determining <area>.

# <detail_info>

# Here are some prediction results:
# <prediction_results>

# Attention: The given some prediction results are for reference only and cannot be regarded as completely correct answers. You still need to output the final answer by judging the Dialog<and KB>.

# Please determine whether the following response contains a <area> based on input Dialog<, KB,> and prediction result. Please give your reasons and output the json format of "Answer: {"output": "yes/no"}" so that it can be parsed by the program.

# Dialog: <dialog>
# KB: <kb>
# '''
    prompt = '''You are an expert at determining <area>.

<detail_info>

Here are some prediction results:
<prediction_results>

The prediction results are for reference only. Please make careful judgments based on these prediction results and Dialog<, KB,> to prevent some Inconsistency from being mistakenly judged as yes.
Please determine whether the following response contains a <area> based on input Dialog<, KB,> and prediction result. Please give your reasons and output the json format of "Answer: {"output": "yes/no"}" so that it can be parsed by the program.

Dialog: <dialog>
KB: <kb>
'''
    prompt = prompt.replace('<area>', area_info[area])
    prompt = prompt.replace('<detail_info>', detail_info[area])
    # prompt = prompt.replace('<attention>', attention_info[area])

    prompt = prompt.replace('<dialog>', dialog)
    if area == 'kbi':
        prompt = prompt.replace('<and KB>', ' and KB')
        prompt = prompt.replace('<, KB,>', ', KB,')
        prompt = prompt.replace('<kb>', kb)
    else:
        prompt = prompt.replace('<and KB>', '')
        prompt = prompt.replace('<, KB,>', '')
        prompt = prompt.replace('KB: <kb>', '')

    prediction_results = ''
    for pred_id in prediction_id:
        prediction_results += '{} prediction result: {}.'.format(area_info[pred_id], prev_list['pred'][pred_id])
        prediction_results += '\n'
    prompt = prompt.replace('<prediction_results>', prediction_results)

    return prompt