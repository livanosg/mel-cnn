#python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 64 -opt adamax -ncd -loss focal -cval -clrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-5 -dor 0.2 # 01
#python main.py -pt effnet6 -task ben_mal -it both -is 224 -btch 64 -opt adamax -ncd -loss focal -ws -cval -clrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-5 -dor 0.2 # 02
#python main.py -pt effnet6 -task nev_mel -it clinic -is 224 -btch 64 -opt adamax -ncd -loss focal -cval -clrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-5 -dor 0.2 # 03
#python main.py -pt effnet6 -task nev_mel -it both -is 224 -btch 64 -opt adamax -ncd -loss focal -ws -cval -clrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-5 -dor 0.2 # 04
#python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 64 -nit -opt adamax -loss focal -cval -clrs 128 -dlrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-5 -dor 0.2 # 05
#python main.py -pt effnet6 -task ben_mal -it both -is 224 -btch 64 -opt adamax -loss focal -ws -cval -clrs 128 -dlrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-5 -dor 0.2 # 06
#python main.py -pt effnet6 -task nev_mel -it clinic -is 224 -btch 64 -nit -opt adamax -loss focal -cval -clrs 128 -dlrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-5 -dor 0.2 # 07
#python main.py -pt effnet6 -task nev_mel -it both -is 224 -btch 64 -opt adamax -loss focal -ws -cval -clrs 128 -dlrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-5 -dor 0.2 # 08
python main.py -pt effnet6 -task ben_mal -it both -is 224 -btch 64 -nit -opt adamax -loss focal -ws -cval -clrs 128 -dlrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-5 -dor 0.2 # 09
python main.py -pt effnet6 -task nev_mel -it both -is 224 -btch 64 -nit -opt adamax -loss focal -ws -cval -clrs 128 -dlrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-5 -dor 0.2 # 10
exit