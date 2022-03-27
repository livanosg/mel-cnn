python main.py -pt effnet6 -task ben_mal -it clinic -is 500 -btch 64 -io -loss focal
python main.py -pt effnet6 -task ben_mal -it both -is 500 -btch 64 -io -loss focal
python main.py -pt effnet6 -task nev_mel -it clinic -is 500 -btch 64 -io -loss focal
python main.py -pt effnet6 -task nev_mel -it both -is 500 -btch 64 -io -loss focal
python main.py -pt effnet6 -task ben_mal -it clinic -is 500 -btch 64 -nit -loss focal
python main.py -pt effnet6 -task ben_mal -it both -is 500 -btch 64 -loss focal
python main.py -pt effnet6 -task nev_mel -it clinic -is 500 -btch 64 -nit -loss focal
python main.py -pt effnet6 -task nev_mel -it both -is 500 -btch 64 -loss focal
python main.py -pt effnet6 -task ben_mal -it both -is 500 -btch 64 -nit -loss focal
python main.py -pt effnet6 -task nev_mel -it both -is 500 -btch 64 -nit -loss focal
