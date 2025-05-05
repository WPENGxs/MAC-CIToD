from openai import OpenAI
import copy

# openai
client_gpt = OpenAI(api_key="openai api key", base_url="https://api.openai.com/v1")

# deepinfra
client_deepinfra = OpenAI(api_key="deepinfra api key", base_url="https://api.deepinfra.com/v1/openai")

BREAK_TIMES_LIMIT = 5

class model():
    def __init__(self, model):
        self.model = model

    def gpt_generator(self, text, history=[]):
        tmp_history = copy.deepcopy(history)
        if tmp_history == []:
            tmp_history = [
                {"role": "system", "content": "you are a helpful assistant"},
                {"role": "user", "content": text},
            ]
        else:
            tmp_history.append({"role": "user", "content": text})
        loop_times = 0
        while True:
            if loop_times > BREAK_TIMES_LIMIT:
                message = 'BREAK_TIMES_LIMIT'
                break
            try:
                response = client_gpt.chat.completions.create(
                model=self.model,
                messages=tmp_history,
                temperature=0.3,
                top_p=1,
                max_tokens=512,
                stream=False)
                message = response.choices[0].message.content
            except Exception:
                message = 'Error'
            if message != 'Error':
                break
            loop_times += 1
        tmp_history.append({"role": "system", "content": message})
        return message, tmp_history

    def deepinfra_generator(self, text, history=[]):
        tmp_history = copy.deepcopy(history)
        if tmp_history == []:
            tmp_history = [
                {"role": "system", "content": "you are a helpful assistant"},
                {"role": "user", "content": text},
            ]
        else:
            tmp_history.append({"role": "user", "content": text})
        loop_times = 0
        while True:
            if loop_times > BREAK_TIMES_LIMIT:
                message = 'BREAK_TIMES_LIMIT'
                break
            try:
                response = client_deepinfra.chat.completions.create(
                model=self.model,
                messages=tmp_history,
                temperature=0.7,
                top_p=0.8,
                max_tokens=512,
                stream=False)
                message = response.choices[0].message.content
            except Exception:
                message = 'Error'
            if message != 'Error':
                break
            loop_times += 1
        tmp_history.append({"role": "system", "content": message})
        return message, tmp_history