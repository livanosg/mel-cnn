import os
import sys

from hyperparameters import HWC_DOM, BATCH_SIZE_RANGE, DROPOUT_LST, RELU_A, OPTIMIZER_LST, LR_LST, MODEL_LST
from train_model import training

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# os.system("unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY")
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

partition = 'local' # sys.argv[1]
run_num = 0
for LR in LR_LST.domain.values:
    for batch_size in BATCH_SIZE_RANGE.domain.values:
        for optimizer in OPTIMIZER_LST.domain.values:
            for HWC in HWC_DOM.domain.values:
                for activation_a in RELU_A.domain.values:
                    for model in MODEL_LST.domain.values:
                        for dropout_rate in (DROPOUT_LST.domain.min_value, DROPOUT_LST.domain.max_value):
                            hparams = {LR_LST: LR,
                                       BATCH_SIZE_RANGE: batch_size,
                                       HWC_DOM: HWC,
                                       RELU_A: activation_a,
                                       DROPOUT_LST: dropout_rate,
                                       OPTIMIZER_LST: optimizer,
                                       MODEL_LST: model}
                            print({h.name: hparams[h] for h in hparams})
                            training(partition=partition, hparams=hparams, log_dir=f'logs/run-{run_num}')
                            run_num += 1
