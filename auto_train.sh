python main.py -pt effnet6 -task nev_mel -it derm -is 224 -btch 32 -opt adamax -ncd -loss focal -clrs 128 -mlrs 512 -l1 0. -l2 1e-7 -dor 0.2 -lr 1e-5 # 3 DONE
#python main.py -pt effnet6 -task nev_mel -it clinic -is 224 -btch 32 -opt adamax -ncd -loss focal -clrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-5 -dor 0.2 # 3 DONE
#python main.py -pt effnet6 -task nev_mel -it clinic -is 224 -btch 32 -opt adamax -ncd -loss focal -clrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-5 -dor 0.3 # 3 DONE
exit
#python main.py -pt effnet6 -task nev_mel -it both -is 224 -btch 32 -opt adamax -ncd -loss focal -cv -clrs 256 -mlrs 256 -l1 0. -l2 0. -lr 1e-4 -dor 0.1 # 4
#python main.py -pt effnet6 -task nev_mel -it both -is 224 -btch 32 -opt adamax -ncd -loss focal -ws -cv -clrs 256 -mlrs 256 -l1 0. -l2 0. -lr 1e-4 -dor 0.1  # 4
#python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 32 -ncd -opt adamax -loss focal -clrs 256 -mlrs 256 -lr 1e-4 -dor 0.3  # 1
#python main.py -pt effnet6 -task ben_mal -it both -is 224 -btch 32 -ncd -opt adamax -loss focal -clrs 256 -mlrs 256 -lr 1e-4 -dor 0.3  # 2
#python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 32 -nit -opt adamax -loss focal -clrs 256 -dlrs 64 -mlrs 256 -lr 1e-4 -dor 0.3 # 5
#python main.py -pt effnet6 -task ben_mal -it both -is 224 -btch 32 -opt adam -loss focal -clrs 256 -dlrs 64 -mlrs 256 -lr 1e-4 -dor 0.3 # 6
#python main.py -pt effnet6 -task nev_mel -it clinic -is 224 -btch 32 -nit -opt adam -loss focal -clrs 256 -dlrs 64 -mlrs 256 -lr 1e-4 -dor 0.3 # 7
#python main.py -pt effnet6 -task nev_mel -it both -is 224 -btch 32 -opt adam -loss focal -clrs 256 -dlrs 64 -mlrs 256 -lr 1e-4 -dor 0.3 # 8
#python main.py -pt effnet6 -task ben_mal -it both -is 224 -btch 32 -nit -opt adam -loss focal -clrs 256 -dlrs 64 -mlrs 256 -lr 1e-4 -dor 0.3 # 9
#python main.py -pt effnet6 -task nev_mel -it both -is 224 -btch 32 -nit -opt adam -loss focal -clrs 256 -dlrs 64 -mlrs 256 -lr 1e-4 -dor 0.3 # 10
