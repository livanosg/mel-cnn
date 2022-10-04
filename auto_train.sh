#LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
#python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 32 -opt adamax -ncd -loss focal -cval -clrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-6 -dor 0.2 -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/clinic/170422073610 # 01 trained on clinic
#python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 32 -opt adamax -ncd -loss focal -ws -cval -clrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-6 -dor 0.2 -test -load /home/giorgos/projects/mel-cnn/models/ben_mal/both/170422084809 # 02 trained on both same as above
#python main.py -pt effnet6 -task nev_mel -it clinic -is 224 -btch 32 -opt adamax -ncd -loss focal -cval -clrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-6 -dor 0.2 -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/clinic/170422161255 # 03 trained on clinic
#python main.py -pt effnet6 -task nev_mel -it clinic -is 224 -btch 32 -opt adamax -ncd -loss focal -ws -cval -clrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-6 -dor 0.2 -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/both/170422170216 # 04 trained on both same as above
#python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 32 -nit -opt adamax -loss focal -cval -clrs 128 -dlrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-6 -dor 0.2 -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/clinic/180422002418 # 05  trained on clinic
#python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 32 -opt adamax -loss focal -ws -cval -clrs 128 -dlrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-6 -dor 0.2 -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/both/180422073824 # 06
#python main.py -pt effnet6 -task nev_mel -it clinic -is 224 -btch 32 -nit -opt adamax -loss focal -cval -clrs 128 -dlrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-6 -dor 0.2 -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/clinic/180422214830 # 07 trained on clinic
#python main.py -pt effnet6 -task nev_mel -it clinic -is 224 -btch 32 -opt adamax -loss focal -ws -cval -clrs 128 -dlrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-6 -dor 0.2 -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/both/180422224234 # 08
#python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 32 -nit -opt adamax -loss focal -ws -cval -clrs 128 -dlrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-6 -dor 0.2 -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/both/190422044340 # 09 trained on both same as 5
#python main.py -pt effnet6 -task nev_mel -it clinic -is 224 -btch 32 -nit -opt adamax -loss focal -ws -cval -clrs 128 -dlrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-6 -dor 0.2 -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/both/190422175945 # 10 trained on both same as 7
#python main.py -pt effnet6 -task nev_mel -it clinic -is 224 -btch 64 -opt adamax -loss focal -ws -cval -clrs 128 -dlrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-5 -dor 0.2
#python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 64 -opt adamax -loss focal -ws -cval -clrs 128 -dlrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-5 -dor 0.2
#python main.py -pt effnet6 -task nev_mel -it clinic -is 224 -btch 32 -opt adamax -loss focal -ws -cval -clrs 128 -dlrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-6 -dor 0.2 -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/clinic/010622060530
#python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 32 -opt adamax -loss focal -ws -cval -clrs 128 -dlrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-6 -dor 0.2 -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/clinic/010622063249
#LD_PRELOAD="/usr/lib/libtcmalloc.so"

#python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/both/090722225549 # all inputs
#python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/both/100722061940 # all inputs
#python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/both/100722120034 # all inputs
#python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/both/100722165740 # all inputs

#python main.py -pt effnet6 -task ben_mal -it clinic -ncd -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/both/110722002904 # only image
#python main.py -pt effnet6 -task ben_mal -it clinic -ncd -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/both/110722084420 # only image
#python main.py -pt effnet6 -task ben_mal -it clinic -ncd -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/both/110722123708 # only image
#python main.py -pt effnet6 -task ben_mal -it clinic -ncd -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/both/110722181754 # only image

#python main.py -pt effnet6 -task ben_mal -it clinic -nit -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/both/120722000345 # all inputs no image type
#python main.py -pt effnet6 -task ben_mal -it clinic -nit -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/both/120722080652 # all inputs no image type
#python main.py -pt effnet6 -task ben_mal -it clinic -nit -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/both/120722160228 # all inputs no image type
python main.py -pt effnet6 -task ben_mal -it clinic -nit -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/both/120722211759 # all inputs no image type

python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/clinic/150622021534 # all inputs
python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/clinic/150622030505 # all inputs
python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/clinic/150622042431 # all inputs
python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/clinic/150622053312 # all inputs

python main.py -pt effnet6 -task ben_mal -it clinic -ncd -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/clinic/150622102931 # only image
python main.py -pt effnet6 -task ben_mal -it clinic -ncd -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/clinic/150622114339 # only image
python main.py -pt effnet6 -task ben_mal -it clinic -ncd -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/clinic/150622122752 # only image
python main.py -pt effnet6 -task ben_mal -it clinic -ncd -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/clinic/150622131841 # only image

python main.py -pt effnet6 -task ben_mal -it clinic -nit -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/clinic/150622063258 # all inputs no image type
python main.py -pt effnet6 -task ben_mal -it clinic -nit -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/clinic/150622072546 # all inputs no image type
python main.py -pt effnet6 -task ben_mal -it clinic -nit -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/clinic/150622082758 # all inputs no image type
python main.py -pt effnet6 -task ben_mal -it clinic -nit -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/ben_mal/clinic/150622094958 # all inputs no image type

python main.py -pt effnet6 -task nev_mel -it clinic -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/both/260622221920 # all inputs
python main.py -pt effnet6 -task nev_mel -it clinic -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/both/270622014519 # all inputs
python main.py -pt effnet6 -task nev_mel -it clinic -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/both/270622052010 # all inputs
python main.py -pt effnet6 -task nev_mel -it clinic -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/both/270622072434 # all inputs

python main.py -pt effnet6 -task nev_mel -it clinic -ncd -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/both/270622114812 # only image
python main.py -pt effnet6 -task nev_mel -it clinic -ncd -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/both/270622140100 # only image
python main.py -pt effnet6 -task nev_mel -it clinic -ncd -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/both/270622160153 # only image
python main.py -pt effnet6 -task nev_mel -it clinic -ncd -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/both/270622180208 # only image

python main.py -pt effnet6 -task nev_mel -it clinic -nit -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/both/270622205346  # all inputs no image type
python main.py -pt effnet6 -task nev_mel -it clinic -nit -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/both/270622224025  # all inputs no image type
python main.py -pt effnet6 -task nev_mel -it clinic -nit -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/both/280622014610  # all inputs no image type
python main.py -pt effnet6 -task nev_mel -it clinic -nit -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/both/280622032236 # all inputs no image type

python main.py -pt effnet6 -task nev_mel -it clinic -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/clinic/140622014301 # all inputs
python main.py -pt effnet6 -task nev_mel -it clinic -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/clinic/140622022833 # all inputs
python main.py -pt effnet6 -task nev_mel -it clinic -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/clinic/140622030628 # all inputs
python main.py -pt effnet6 -task nev_mel -it clinic -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/clinic/140622034105 # all inputs

python main.py -pt effnet6 -task nev_mel -it clinic -ncd -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/clinic/140622040912 # only image
python main.py -pt effnet6 -task nev_mel -it clinic -ncd -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/clinic/140622042044 # only image
python main.py -pt effnet6 -task nev_mel -it clinic -ncd -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/clinic/150622000657 # only image
python main.py -pt effnet6 -task nev_mel -it clinic -ncd -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/clinic/150622002915 # only image

python main.py -pt effnet6 -task nev_mel -it clinic -nit -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/clinic/150622004812 # all inputs no image type
python main.py -pt effnet6 -task nev_mel -it clinic -nit -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/clinic/150622011110 # all inputs no image type
python main.py -pt effnet6 -task nev_mel -it clinic -nit -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/clinic/150622012621 # all inputs no image type
python main.py -pt effnet6 -task nev_mel -it clinic -nit -is 224 -btch 32 -lr 1e-6 -ws -cval -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/clinic/150622014331 # all inputs no image type

#python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 64 -ws -cval # all inputs
#python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 64 -ws -cval # all inputs
#python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 64 -ws -cval # all inputs
#
#python main.py -pt effnet6 -task ben_mal -it clinic -nit -is 224 -btch 64 -ws -cval # all inputs no image type
#python main.py -pt effnet6 -task ben_mal -it clinic -nit -is 224 -btch 64 -ws -cval # all inputs no image type
#python main.py -pt effnet6 -task ben_mal -it clinic -nit -is 224 -btch 64 -ws -cval # all inputs no image type
#python main.py -pt effnet6 -task ben_mal -it clinic -nit -is 224 -btch 64 -ws -cval # all inputs no image type
#
#python main.py -pt effnet6 -task ben_mal -it clinic -ncd -is 224 -btch 64 -ws -cval # only image
#python main.py -pt effnet6 -task ben_mal -it clinic -ncd -is 224 -btch 64 -ws -cval # only image
#python main.py -pt effnet6 -task ben_mal -it clinic -ncd -is 224 -btch 64 -ws -cval # only image
#python main.py -pt effnet6 -task ben_mal -it clinic -ncd -is 224 -btch 64 -ws -cval # only image

#python main.py -pt effnet6 -task nev_mel -it clinic -is 224 -btch 32 -opt adamax -ncd -loss focal -ws -cval -clrs 128 -mlrs 512 -l1 0. -l2 1e-7 -lr 1e-6 -dor 0.2 -fine -load /home/giorgos/projects/mel-cnn/models/nev_mel/both/020622023230 # fine tune above
