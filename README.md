# vovnet_mindspore

由于`vovnet`有不同的参数量的模型，所以先尝试了一个轻量级的小模型进行前向过程。

`VoVNet19_eSE_mindspore.py`里面还有一个关于层名称的报错还无法解决（已解决）,但是在`VoVNet19_eSE_pytorch.py`里面却没有任何问题。

还有一个算子还没有完成开发，是关于冻结BN层参数的问题。

出现了一个新的问题，原代码中继承`nn.Sequential`类的代码没有写`forward`函数，但是在`mindspore`中却无法输出正确的值。
