# MRAT

This is the source code for the paper `Robust Neural Text Classification and Entailment via Mixup
Regularized Adversarial Training` (MRAT).

## Setup

1. Create environment and install requirement packages using provided `environment.yml`:

   ```
   conda env create -f environment.yml
   conda activate MRAT
   ```

2. Download datasets. We follow the practice of [textfooler](https://github.com/jind11/TextFooler) and use the dataset they provided.

   * Download `mr.zip` and `snli.zip` datasets from [googledrive](https://drive.google.com/drive/folders/1N-FYUa5XN8qDs4SgttQQnrkeTXXAXjTv).
   * Extract `mr.zip` to `train/data/mr` and extract `snli.zip` to `train/data/snli` .
   * Download 1k split datasets of `mr` and `snli`  from [url](https://github.com/jind11/TextFooler/tree/master/data).
   * Put `mr` and `snli` to `attack/data` .

* [Optional] For ease of replication, we shared `adversarial examples` and `trained BERT model` on each dataset we used. Details see [here](outputs/readme.md).

## Usage

1. Train victim model. Alternatively, you can download our trained victim model from [here](outputs/readme.md).

   ```
   cd train/train_command_mr/
   python bert_mr_normal.py
   ```

   > For `snli` dataset we do not need this step. We use the `bert-base-uncased-snli` from [TextAttack Model Zoo](https://textattack.readthedocs.io/en/latest/3recipes/models.html) as attack target model on snli dataset. 

2. Test the adversarial robustness of the victim model.

   ```
   cd ../../
   cd attack/attack_command_mr/
   python attack_bert_mr_test_textbugger.py
   python attack_bert_mr_test_deepwordbug.py
   python attack_bert_mr_test_textfooler.py
   ```

3. Attack victim model on training set and save generate adversarial examples. Alternatively, you can download our generated `adversarial examples` from [here](outputs/readme.md).

   ```
   python attack_bert_mr_train_textbugger.py
   python attack_bert_mr_train_deepwordbug.py
   python attack_bert_mr_train_textfooler.py
   ```

4. Train `MRAT` and `MRAT+` models. Alternatively,  you can download our trained models from [here](outputs/readme.md).

   ```
   cd ../../
   cd train/train_command_mr/
   python bert_mr_mix_multi.py   # MRAT
   python bert_mr_mixN_multi.py   # MRAT+
   ```

5. Test the adversarial robustness of `MRAT` and `MRAT+`.

   ```
   cd ../../
   cd attack/attack_command_mr/
   # MRAT
   python attack_bert_mr_test_mix-multi_textbugger.py
   python attack_bert_mr_test_mix-multi_deepwordbug.py
   python attack_bert_mr_test_mix-multi_textfooler.py
   # MRAT+
   python attack_bert_mr_test_mixN-multi_textbugger.py
   python attack_bert_mr_test_mixN-multi_deepwordbug.py
   python attack_bert_mr_test_mixN-multi_textfooler.py
   ```

* Above steps uses `mr` dataset as example. For `snli` dataset, the usage is the same. Just change `attack_command_mr` and `train_command_mr` to `attack_command_snli` and `train_command_snli` folds and run the python scripts inside. 
* Follows the above steps, you can reproduce the results in `Table 2` of our paper. For additional results, a little change of argument setting in this scripts may be needed. 
* The code has been tested on CentOS 7 using multi-GPU. For single GPU usage, a little change may be needed. 
* If you need helps, feel free to open an issue. 😊

## Acknowledgement

🎉A huge thanks to [TextAttack](https://github.com/QData/TextAttack)!~  The most of the code in this project is derived from the helpful toolbox TextAttack. Because TextAttack update frequently, we leave a copy in `attack\textattack` fold.

