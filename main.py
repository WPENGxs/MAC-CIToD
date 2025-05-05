from model import model
from os_model import os_model
from mac_citod import mac_citod
import argparse

# connection = ['full', 'cycle', 'central']
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='mac_citod')
parser.add_argument('--connection', type=str, default='full')
parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')

def main():
    args = parser.parse_args()

    if args.model_name == 'gpt-3.5-turbo':
        eval_model = model('gpt-3.5-turbo')
        generator = eval_model.gpt_generator
    elif args.model_name == 'gpt-4o':
        eval_model = model('gpt-4o')
        generator = eval_model.gpt_generator
    elif args.model_name == 'llama':
        eval_model = model('meta-llama/Meta-Llama-3.1-8B-Instruct')
        generator = eval_model.deepinfra_generator
        # generator = os_model.llama3_generator
    elif args.model_name == 'glm4':
        generator = os_model.glm4_generator
    elif args.model_name == 'gemma':
        eval_model = model('google/gemma-2-9b-it')
        generator = eval_model.deepinfra_generator
    
    m = mac_citod(generator, './log', args.model_name)
    if args.method == 'mac_citod':
        output = m.test_mac_citod(args.connection)
        
    print(output)

if __name__ == '__main__':
    main()