- 官方的数据
kinetics_train.csv
kinetics_val.csv

kinetics-400视频的数据：
kinetics-400-train.txt
kinetics-400-val.txt

- 这两个是我们的kinetics处理成frame的数据：
train_videofolder.txt
val_videofolder.txt
missing两个文件上是相对于官方发布的缺少的视频。

其中 kinetics-400-map.txt 是我们现在数据的label；（注意，这个和官网的不一致，主要在于命名的格式上）另外，我们现在的数据都是经过编码以后的;短边基本为256；

具体数据位置会后续传到didi或者ali的存储上



2021.07.08
- 更新

原先利用TSM官方提供的kinetics的csv下载文件和map去生成帧的训练视频，存在一个问题就是label对不齐，就是label和video的真实label是错的；然后导致loss最后非常的奇怪；

更新的版本是将label都用的新的代码进行生成；
新版本为:
- data/kinetics400/new_train_videofolder.txt (234618/234619, missing 5843)
- data/kinetics400/new_train_videofolder.txt (/19760, missing 493)
删除老版本错误的list