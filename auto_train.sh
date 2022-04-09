#python main.py -pt effnet6 -task nev_mel -it clinic -is 224 -btch 32 -opt adamax -ncd -loss focal -clrs 256 -mlrs 256 -lr 1e-4 -dor 0.3  # 3 DONE
python main.py -pt effnet6 -task nev_mel -it both -is 224 -btch 32 -opt adamax -ncd -loss focal -clrs 256 -mlrs 256 -lr 1e-4 -dor 0.2  # 4
#python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 32 -ncd -opt adamax -loss focal -clrs 256 -mlrs 256 -lr 1e-4 -dor 0.3  # 1
#python main.py -pt effnet6 -task ben_mal -it both -is 224 -btch 32 -ncd -opt adamax -loss focal -clrs 256 -mlrs 256 -lr 1e-4 -dor 0.3  # 2

#python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 32 -nit -opt adamax -loss focal -clrs 256 -dlrs 64 -mlrs 256 -lr 1e-4 -dor 0.3 # 5
#python main.py -pt effnet6 -task ben_mal -it both -is 224 -btch 32 -opt adam -loss focal -clrs 256 -dlrs 64 -mlrs 256 -lr 1e-4 -dor 0.3 # 6
#python main.py -pt effnet6 -task nev_mel -it clinic -is 224 -btch 32 -nit -opt adam -loss focal -clrs 256 -dlrs 64 -mlrs 256 -lr 1e-4 -dor 0.3 # 7
#python main.py -pt effnet6 -task nev_mel -it both -is 224 -btch 32 -opt adam -loss focal -clrs 256 -dlrs 64 -mlrs 256 -lr 1e-4 -dor 0.3 # 8
#python main.py -pt effnet6 -task ben_mal -it both -is 224 -btch 32 -nit -opt adam -loss focal -clrs 256 -dlrs 64 -mlrs 256 -lr 1e-4 -dor 0.3 # 9
#python main.py -pt effnet6 -task nev_mel -it both -is 224 -btch 32 -nit -opt adam -loss focal -clrs 256 -dlrs 64 -mlrs 256 -lr 1e-4 -dor 0.3 # 10
#Epoch 37/500
#14/14 [==============================] - 12s 758ms/step - loss: 0.1616 - f1_macro: 0.7179 - gmean: 0.7255 - val_loss: 0.1042 - val_f1_macro: 0.5449 - val_gmean: 0.4587