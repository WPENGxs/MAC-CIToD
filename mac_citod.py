import json
from tqdm import tqdm
import re
import requests
import os
from prompt import *
from evaluate import *

class mac_citod():
    def __init__(self, model, log_path, model_name):
        self.model = model
        self.log_path = f'{log_path}'
        self.model_name = model_name

    def check_dir(self, path):
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except OSError as e:
                print(f"An error occurred while creating path '{path}': {e.strerror}")

    def get_predict_ans(self, text):
        predict_label = -1
        text = str(text)
        answer = re.search(r'(?<="output": ")[^"]+', text)
        if answer:
            answer = answer.group()
            if answer == 'yes' or answer == 'Yes':
                predict_label = 1
            elif answer == 'no' or answer == 'No':
                predict_label = 0
            else:
                predict_label = -1

        return predict_label

    def get_json_ans(self, text, obj):
        pattern = rf'(?<="{obj}": ")[^"]+'
        answer = re.search(pattern, text)
        if answer:
            answer = answer.group()
            if answer == 'yes' or answer == 'Yes':
                predict_label = 1
            elif answer == 'no' or answer == 'No':
                predict_label = 0
            else:
                predict_label = -1
        else:
            predict_label = -1
        return predict_label

    def rm_neg_num(self, predict_label, gold_label):
        if predict_label == -1:
            if gold_label == 1:
                return 0
            elif gold_label == 0:
                return 1
        else:
            return predict_label

    def get_pre_area_ans(self, pred):
        if pred == 1:
            return 'yes'
        elif pred == 0:
            return 'no'
        elif pred == -1:
            return 'unknown'

    def _get_dataloader(self, data_path):
        data = []

        load_list = ['calendar', 'navigate','weather_new']
        for l in load_list:
            with open(f'{data_path}/{l}_test.json', 'r') as json_file:
                json_ = json.load(json_file)
            
            for j in json_:
                d = {}
                d['dialogue'] = j['dialogue']
                d['kb'] = j['scenario']['kb']
                qi_gold = j['scenario']['qi']
                hi_gold = j['scenario']['hi']
                kbi_gold = j['scenario']['kbi']
                d['gold'] = [int(qi_gold), int(hi_gold), int(kbi_gold)]
                data.append(d)

        return data
    
    def test_mac_citod(self, network):
        saved_path = f'{self.log_path}/{self.model_name}'
        self.check_dir(saved_path)
        output_json = []
        data = self._get_dataloader('./data')
        pred_1 = []
        pred_2 = []
        gold = []

        with tqdm(total=len(data)) as pbar:
            for d in data:
                output = {
                    'first_round':{
                        'qi':[],
                        'hi':[],
                        'kbi':[],
                    },
                    'first_round_pred':{
                        'qi':-1,
                        'hi':-1,
                        'kbi':-1,
                    },
                    'second_round':{
                        'qi':[],
                        'hi':[],
                        'kbi':[],
                    },
                    'second_round_pred':{
                        'qi':-1,
                        'hi':-1,
                        'kbi':-1,
                    },
                }
                dialog = d['dialogue']
                dialog_qi = dialog[-2:]
                dialog_hi = dialog[0:-2] + dialog[-1:]
                dialog_kbi = dialog[-1:]
                kb = d['kb']
                g = d['gold']
                ##### first round #####
                prompt = get_first_round_prompt('qi', str(dialog), str(kb))
                qi_output, history = self.model(prompt)
                output['first_round']['qi'] = history
                qi_pred = self.get_predict_ans(qi_output)
                output['first_round_pred']['qi'] = qi_pred

                prompt = get_first_round_prompt('hi', str(dialog_hi), str(kb))
                hi_output, history = self.model(prompt)
                output['first_round']['hi'] = history
                hi_pred = self.get_predict_ans(hi_output)
                output['first_round_pred']['hi'] = hi_pred

                prompt = get_first_round_prompt('kbi', str(dialog_kbi), str(kb))
                kbi_output, history = self.model(prompt)
                output['first_round']['kbi'] = history
                kbi_pred = self.get_predict_ans(kbi_output)
                output['first_round_pred']['kbi'] = kbi_pred

                pred_1.append([self.rm_neg_num(qi_pred, g[0]), self.rm_neg_num(hi_pred, g[1]), self.rm_neg_num(kbi_pred, g[2])])
                ##### second round #####
                prev_list = {
                    'reason': {
                        'qi': qi_output,
                        'hi': hi_output,
                        'kbi': kbi_output
                    },
                    'pred': {
                        'qi': self.get_pre_area_ans(qi_pred),
                        'hi': self.get_pre_area_ans(hi_pred),
                        'kbi': self.get_pre_area_ans(kbi_pred)
                    }
                }
                if network == 'full': 
                    ##### full #####
                    prediction_id = ['qi', 'hi', 'kbi']
                    prompt = get_second_round_prompt('qi', str(dialog), str(kb), prev_list, prediction_id)
                    qi_output, history = self.model(prompt)
                    output['second_round']['qi'] = history
                    qi_pred = self.get_predict_ans(qi_output)
                    output['second_round_pred']['qi'] = qi_pred

                    prompt = get_second_round_prompt('hi', str(dialog_hi), str(kb), prev_list, prediction_id)
                    hi_output, history = self.model(prompt)
                    output['second_round']['hi'] = history
                    hi_pred = self.get_predict_ans(hi_output)
                    output['second_round_pred']['hi'] = hi_pred

                    prompt = get_second_round_prompt('kbi', str(dialog_kbi), str(kb), prev_list, prediction_id)
                    kbi_output, history = self.model(prompt)
                    output['second_round']['kbi'] = history
                    kbi_pred = self.get_predict_ans(kbi_output)
                    output['second_round_pred']['kbi'] = kbi_pred
                elif network == 'cycle':
                    ##### cycle #####
                    prediction_id = ['kbi']
                    prompt = get_second_round_prompt('qi', str(dialog), str(kb), prev_list, prediction_id)
                    qi_output, history = self.model(prompt)
                    output['second_round']['qi'] = history
                    qi_pred = self.get_predict_ans(qi_output)
                    output['second_round_pred']['qi'] = qi_pred

                    prediction_id = ['qi']
                    prompt = get_second_round_prompt('hi', str(dialog_hi), str(kb), prev_list, prediction_id)
                    hi_output, history = self.model(prompt)
                    output['second_round']['hi'] = history
                    hi_pred = self.get_predict_ans(hi_output)
                    output['second_round_pred']['hi'] = hi_pred

                    prediction_id = ['hi']
                    prompt = get_second_round_prompt('kbi', str(dialog_kbi), str(kb), prev_list, prediction_id)
                    kbi_output, history = self.model(prompt)
                    output['second_round']['kbi'] = history
                    kbi_pred = self.get_predict_ans(kbi_output)
                    output['second_round_pred']['kbi'] = kbi_pred
                elif network == 'central':
                    ##### central #####
                    prediction_id = ['kbi']
                    prompt = get_second_round_prompt('qi', str(dialog), str(kb), prev_list, prediction_id)
                    qi_output, history = self.model(prompt)
                    output['second_round']['qi'] = history
                    qi_pred = self.get_predict_ans(qi_output)
                    output['second_round_pred']['qi'] = qi_pred

                    prediction_id = ['qi', 'kbi']
                    prompt = get_second_round_prompt('hi', str(dialog_hi), str(kb), prev_list, prediction_id)
                    hi_output, history = self.model(prompt)
                    output['second_round']['hi'] = history
                    hi_pred = self.get_predict_ans(hi_output)
                    output['second_round_pred']['hi'] = hi_pred

                    prediction_id = ['qi']
                    prompt = get_second_round_prompt('kbi', str(dialog_kbi), str(kb), prev_list, prediction_id)
                    kbi_output, history = self.model(prompt)
                    output['second_round']['kbi'] = history
                    kbi_pred = self.get_predict_ans(kbi_output)
                    output['second_round_pred']['kbi'] = kbi_pred

                pred_2.append([self.rm_neg_num(qi_pred, g[0]), self.rm_neg_num(hi_pred, g[1]), self.rm_neg_num(kbi_pred, g[2])])

                gold.append(g)
                output_json.append(output)
                pbar.update(1)

                # break

        output = {
            "first_round_eval": {},
            "second_round_eval": {},
        }
        output['first_round_eval'] = EvaluateTool.evaluate(pred_1, gold)
        output['second_round_eval'] = EvaluateTool.evaluate(pred_2, gold)
        output_json.append(output)

        output_json = json.dumps(output_json)
        file_path = f'{saved_path}/output_{network}.json'
        output_file = open(file_path, 'w', encoding='utf-8')
        output_file.write(output_json)
        output_file.close()

        return output