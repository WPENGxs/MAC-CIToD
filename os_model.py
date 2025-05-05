BREAK_TIMES_LIMIT = 5

class os_model():
    cached_models = {}
    cached_tokenizers = {}

    def mistral_generator(text,history=[]):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
        if 'mistral' not in model.cached_models:
            model_name = "mistral"

            mistral_model = LLM(
                model=model_name
            )

            model.cached_models['mistral'] = mistral_model
        else:
            mistral_model = model.cached_models['mistral']
        if history == []:
            history = [
                {"role": "system", "content": 'you are a helpful assistant'},
                {"role": "user", "content": text},
            ]
        else:
            history = history.append({"role": "user", "content": text})
        try:
            res = mistral_model.chat(messages=history, sampling_params=sampling_params)
            message = res[0].outputs[0].text
        except Exception as e:
            print(e)
            import sys
            sys.exit(1)
            message = "Error"

        history.append({"role": "system", "content": message})
        return message, history

    def llama3_generator(text, history=[], language='en', mode='direct'):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        # Model initialization inside the method with caching
        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
        if 'llama3' not in model.cached_models:
            model_name = "llama-3.1"

            llama3_model = LLM(model=model_name)
            llama3_tokenizer = AutoTokenizer.from_pretrained(model_name)

            model.cached_models['llama3'] = llama3_model
            model.cached_tokenizers['llama3'] = llama3_tokenizer
        else:
            llama3_model = model.cached_models['llama3']
            llama3_tokenizer = model.cached_tokenizers['llama3']
            # device = llama3_model.device
        if history == []:
            history = [
                {"role": "system", "content": 'you are a helpful assistant'},
                {"role": "user", "content": text},
            ]
        else:
            history = history.append({"role": "user", "content": text})
        try:
            text_input = llama3_tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True
            )
            outputs = llama3_model.generate([text_input], sampling_params)
            # Print the outputs.
            message = ""
            for output in outputs:
                message += output.outputs[0].text
        except Exception as e:
            print(e)
            import sys
            sys.exit(1)
            message = "Error"

        history.append({"role": "system", "content": message})
        return message, history

    def glm4_generator(text, history=[], language='en', mode='direct'):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        stop_token_ids = [151329, 151336, 151338]
        # Model initialization inside the method with caching
        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512,stop_token_ids = stop_token_ids)
        if 'glm4' not in model.cached_models:
            model_name = "glm-4-9b-chat"

            glm4_model = LLM(
                model=model_name,
                tensor_parallel_size=1,
                max_model_len=131072,
                trust_remote_code=True,
                enforce_eager=True,
                # GLM-4-9B-Chat-1M 如果遇见 OOM 现象，建议开启下述参数
                # enable_chunked_prefill=True,
                # max_num_batched_tokens=8192
            )
            glm4_tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)

            model.cached_models['glm4'] = glm4_model
            model.cached_tokenizers['glm4'] = glm4_tokenizer
        else:
            glm4_model = model.cached_models['glm4']
            glm4_tokenizer = model.cached_tokenizers['glm4']
            # device = llama3_model.device
        if history == []:
            history = [
                {"role": "system", "content": 'you are a helpful assistant'},
                {"role": "user", "content": text},
            ]
        else:
            history = history.append({"role": "user", "content": text})
        try:
            text_input = glm4_tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True
            )
            outputs = glm4_model.generate([text_input], sampling_params)
            # Print the outputs.
            message = ""
            for output in outputs:
                message += output.outputs[0].text
        except Exception as e:
            print(e)
            import sys
            sys.exit(1)
            message = "Error"

        history.append({"role": "system", "content": message})
        return message, history