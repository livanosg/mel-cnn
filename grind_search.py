import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from hyperparameters import HWC_RNG, BATCH_SIZE_RANGE, DROPOUT_LST, RELU_A_LST, OPTIMIZER_LST, LR_LST, MODEL_LST
from train_model import training

run_num = 0
for LR in LR_LST.domain.values:
    for batch_size in BATCH_SIZE_RANGE.domain.values:
        for optimizer in OPTIMIZER_LST.domain.values:
            for HWC in HWC_RNG.domain.values:
                for activation_a in RELU_A_LST.domain.values:
                    for model in MODEL_LST.domain.values:
                        for droput_rate in (DROPOUT_LST.domain.min_value, DROPOUT_LST.domain.max_value):
                            hparams = {LR_LST: LR,
                                       BATCH_SIZE_RANGE: batch_size,
                                       HWC_RNG: HWC,
                                       RELU_A_LST: activation_a,
                                       DROPOUT_LST: droput_rate,
                                       OPTIMIZER_LST: optimizer,
                                       MODEL_LST: model}
                            print({h.name: hparams[h] for h in hparams})
                            training(hparams, log_dir=f'logs/run-{run_num}')
                            run_num += 1
