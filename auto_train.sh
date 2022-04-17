python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 128 -opt adamax -ncd -loss focal -cval -clrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-4 -dor 0.2  # 1
python main.py -pt effnet6 -task ben_mal -it both -is 224 -btch 128 -opt adamax -ncd -loss focal -ws -cval -clrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-4 -dor 0.2  # 2
python main.py -pt effnet6 -task nev_mel -it clinic -is 224 -btch 128 -opt adamax -ncd -loss focal -cval -clrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-4 -dor 0.2 # 3
python main.py -pt effnet6 -task nev_mel -it both -is 224 -btch 128 -opt adamax -ncd -loss focal -ws -cval -clrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-4 -dor 0.2  # 4
python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 128 -nit -opt adamax -loss focal -cval -clrs 128 -dlrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-4 -dor 0.2 # 5
python main.py -pt effnet6 -task ben_mal -it both -is 224 -btch 128 -opt adamax -loss focal -ws -cval -clrs 128 -dlrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-4 -dor 0.2 # 6
python main.py -pt effnet6 -task nev_mel -it clinic -is 224 -btch 128 -nit -opt adamax -loss focal -cval -clrs 128 -dlrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-4 -dor 0.2 # 7
python main.py -pt effnet6 -task nev_mel -it both -is 224 -btch 128 -opt adamax -loss focal -ws -cval -clrs 128 -dlrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-4 -dor 0.2 # 8
python main.py -pt effnet6 -task ben_mal -it both -is 224 -btch 128 -nit -opt adamax -loss focal -ws -cval -clrs 128 -dlrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-4 -dor 0.2 # 9
python main.py -pt effnet6 -task nev_mel -it both -is 224 -btch 128 -nit -opt adamax -loss focal -ws -cval -clrs 128 -dlrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-4 -dor 0.2 # 10
exit