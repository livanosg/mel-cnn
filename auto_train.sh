python main.py -pt effnet6 -task nev_mel -it clinic -is 224 -btch 64 -opt adamax -ncd -loss focal -clrs 128 -mlrs 128 -dor 0.2 -lr 1e-5 -strg mirrored -gpus 2 # 3
#python main.py -pt effnet6 -task nev_mel -it clinic -is 224 -btch 64 -opt adamax -ncd -loss focal -wl -clrs 128 -mlrs 128 -dor 0.2 -lr 1e-5 -strg mirrored -gpus 2 # 3
#python main.py -pt effnet6 -task nev_mel -it both -is 224 -btch 64 -opt adamax -ws -loss focal -clrs 128 -mlrs 128 -dor 0.3 -lr 1e-5 -strg mirrored -gpus 2 # 3
#python main.py -pt effnet6 -task nev_mel -it both -is 224 -btch 64 -ncd -opt adam -loss focal -clrs 128 -mlrs 128 -dor 0.3 -lr 1e-4 -strg mirrored -gpus 2 # 4
#python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 64 -ncd -opt adam -loss focal -lossf 1. -clrs 128 -mlrs 128 -dor 0.3 -lr 1e-5 -strg mirrored -gpus 2 # 1
#python main.py -pt effnet6 -task ben_mal -it both -is 224 -btch 64 -ncd -opt adam -loss focal -clrs 128 -mlrs 128 -dor 0.3 -lr 1e-5 -strg mirrored -gpus 2 # 2
#python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 64 -nit -opt adam -loss focal -clrs 128 -dlrs 64 -mlrs 128 -dor 0.3 -lr 1e-4 -strg mirrored -gpus 2 # 5
#python main.py -pt effnet6 -task ben_mal -it both -is 224 -btch 64 -opt adam -loss focal -clrs 128 -dlrs 64 -mlrs 128 -dor 0.3 -lr 1e-4 -strg mirrored -gpus 2 # 6
#python main.py -pt effnet6 -task nev_mel -it clinic -is 224 -btch 64 -nit -opt adam -loss focal -clrs 128 -dlrs 64 -mlrs 128 -dor 0.3 -lr 1e-4 -strg mirrored -gpus 2 # 7
#python main.py -pt effnet6 -task nev_mel -it both -is 224 -btch 64 -opt adam -loss focal -clrs 128 -dlrs 64 -mlrs 128 -dor 0.3 -lr 1e-4 -strg mirrored -gpus 2 # 8
#python main.py -pt effnet6 -task ben_mal -it both -is 224 -btch 64 -nit -opt adam -loss focal -clrs 128 -dlrs 64 -mlrs 128 -dor 0.3 -lr 1e-4 -strg mirrored -gpus 2 # 9
#python main.py -pt effnet6 -task nev_mel -it both -is 224 -btch 64 -nit -opt adam -loss focal -clrs 128 -dlrs 64 -mlrs 128 -dor 0.3 -lr 1e-4 -strg mirrored -gpus 2 # 10


#7/7 [==============================] - 60s 6s/step - loss: 0.1061 - gmean_sens_spec: 0.4426 - f1_score: 0.5670 - val_loss: 0.2162 - val_gmean_sens_spec: 0.1348 - val_f1_score: 0.0838 - lr: 2.0000e-04
#Epoch 2/500
#7/7 [==============================] - 20s 3s/step - loss: 0.0541 - gmean_sens_spec: 0.5544 - f1_score: 0.6587 - val_loss: 0.0498 - val_gmean_sens_spec: 0.4799 - val_f1_score: 0.7425 - lr: 2.0000e-04

