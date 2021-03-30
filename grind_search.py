from hyperparameters import HWC_RANGE, BATCH_SIZE_RANGE, DROPOUT_RANGE, ACTIVATION_OPTIONS, HP_OPTIMIZER, LEARNING_RATE_RANGE, HP_MODELS
from train_model import training

run_num = 0
for LR in LEARNING_RATE_RANGE.domain.values:
    for batch_size in BATCH_SIZE_RANGE.domain.values:
        for optimizer in HP_OPTIMIZER.domain.values:
            for HWC in HWC_RANGE.domain.values:
                for activation_a in ACTIVATION_OPTIONS.domain.values:
                    for model in HP_MODELS.domain.values:
                        for droput_rate in (DROPOUT_RANGE.domain.min_value, DROPOUT_RANGE.domain.max_value):
                            hparams = {LEARNING_RATE_RANGE: LR,
                                       BATCH_SIZE_RANGE: batch_size,
                                       HWC_RANGE: HWC,
                                       ACTIVATION_OPTIONS: activation_a,
                                       DROPOUT_RANGE: droput_rate,
                                       HP_OPTIMIZER: optimizer,
                                       HP_MODELS: model}
                            print({h.name: hparams[h] for h in hparams})
                            training(hparams, log_dir=f'logs/run-{run_num}')
                            run_num += 1
