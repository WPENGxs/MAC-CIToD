<h2 align="center">Improving Consistency Identification in Task-oriented Dialogue through Multi-Agent Collaboration</h2>

<p align="center">
  <b>
  [<a href="paper_url">Paper</a>]
  </b>
  <br/>
</p>

### Table of Contents
- [MAC-CIToD](#mac-citod)
- [Setup](#setup)
- [Performance](#performance)
- [Reference](#reference)
- [Contact](#contact)

## MAC-CIToD
![The main framework of MAC-CIToD](img/main.png)

## Setup
1. Create conda environment:
```bash
conda create -n mac_citod python=3.10
conda activate mac_citod
```

2. Install environment:
```bash
pip install openai tqdm scikit-learn
```

3. (Optional) If you want to run a model based on the API platform, please configure the API key of the corresponding platform in `model.py`:
```python
# openai, including gpt-3.5-turbo, gpt-4o
--> client_gpt = OpenAI(api_key="openai api key", base_url="https://api.openai.com/v1")

# deepinfra, including llama-3.1, gemma-2, 
--> client_deepinfra = OpenAI(api_key="deepinfra api key", base_url="https://api.deepinfra.com/v1/openai")
```

4. (Optional) If you want to run the model locally, please configure the environment and modify the code in main.py as required by the following model:
```bash
For llama-3.1:
pip install

For glm4:
pip install
```

5. Run our code:
```bash
python main.py --connection CONNECTION --model_name MODEL_NAME

CONNECTION = ['full', 'cycle', 'central']
MODEL_NAME = ['gpt-3.5-turbo', 'gpt-4o', 'llama', 'glm4', 'gemma']
```

6. Output final evaluation. After the run is completed, the evaluation code will be run and the corresponding metrics will be output:
```json
{
    "first_round_eval": {
        "precision_qi": 1.0,
        "precision_hi": 1.0,
        "precision_kbi": 1.0,
        "recall_qi": 1.0,
        "recall_hi": 1.0,
        "recall_kbi": 1.0,
        "f1_qi": 1.0,
        "f1_hi": 1.0,
        "f1_kbi": 1.0,
        "overall_acc": 1.0
    },
    "second_round_eval": {
        "precision_qi": 1.0,
        "precision_hi": 1.0,
        "precision_kbi": 1.0,
        "recall_qi": 1.0,
        "recall_hi": 1.0,
        "recall_kbi": 1.0,
        "f1_qi": 1.0,
        "f1_hi": 1.0,
        "f1_kbi": 1.0,
        "overall_acc": 1.0
    }
}
```
In addition, you can get output log in `./log/MODEL_NAME/output_CONNECTION.json`.

## Performance
![Main results](img/performance.png)

## Reference
If you find this project useful for your research, please consider citing the following paper:
```
@article
```

## Contact
If you have any questions or suggestions, please create Github issues here or email [Peng Wang](mailto:wpengxss@gmail.com), and [Libo Qin](mailto:lbqin@csu.edu.cn).