1.更改dataset.py, 更改为子目录为类别，子目录下放置图片
2.更改dataset.py, 先读取labels 生成类别，然后先采样类别，再采样该类别中的图片。
3.更改dataset.py, 采样类别时，采用顺序方式，采样10组相似pair，采样10组不相似pair
4.更改dataset.py, 采样10组10相似pair时，挑选batch中表现最差的10个图片
5.更改dataset.py, 每个epoch前，类别顺序random下

6.需要新建一个验证集文件夹，做交叉验证