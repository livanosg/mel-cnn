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
#MALLOC_TRIM_THRESHOLD=0 with standard malloc
LD_PRELOAD="/usr/lib/libtcmalloc.so"  python main.py -pt effnet6 -task ben_mal -it both -is 224 -btch 64 -ws -cval # all inputs
LD_PRELOAD="/usr/lib/libtcmalloc.so" python main.py -pt effnet6 -task ben_mal -it both -is 224 -btch 64 -ws -cval # all inputs
LD_PRELOAD="/usr/lib/libtcmalloc.so" python main.py -pt effnet6 -task ben_mal -it both -is 224 -btch 64 -ws -cval # all inputs
LD_PRELOAD="/usr/lib/libtcmalloc.so" python main.py -pt effnet6 -task ben_mal -it both -is 224 -btch 64 -ws -cval # all inputs

LD_PRELOAD="/usr/lib/libtcmalloc.so" python main.py -pt effnet6 -task ben_mal -it both -ncd -is 224 -btch 64 -ws -cval # only image
LD_PRELOAD="/usr/lib/libtcmalloc.so" python main.py -pt effnet6 -task ben_mal -it both -ncd -is 224 -btch 64 -ws -cval # only image
LD_PRELOAD="/usr/lib/libtcmalloc.so" python main.py -pt effnet6 -task ben_mal -it both -ncd -is 224 -btch 64 -ws -cval # only image
LD_PRELOAD="/usr/lib/libtcmalloc.so" python main.py -pt effnet6 -task ben_mal -it both -ncd -is 224 -btch 64 -ws -cval # only image

LD_PRELOAD="/usr/lib/libtcmalloc.so" python main.py -pt effnet6 -task ben_mal -it both -nit -is 224 -btch 64 -ws -cval # all inputs no image type
LD_PRELOAD="/usr/lib/libtcmalloc.so" python main.py -pt effnet6 -task ben_mal -it both -nit -is 224 -btch 64 -ws -cval # all inputs no image type
LD_PRELOAD="/usr/lib/libtcmalloc.so" python main.py -pt effnet6 -task ben_mal -it both -nit -is 224 -btch 64 -ws -cval # all inputs no image type
LD_PRELOAD="/usr/lib/libtcmalloc.so" python main.py -pt effnet6 -task ben_mal -it both -nit -is 224 -btch 64 -ws -cval # all inputs no image type

#MALLOC_TRIM_THRESHOLD_=0 python main.py -pt effnet6 -task ben_mal -it clinic -is 224 -btch 64 -ws -cval # all inputs
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
