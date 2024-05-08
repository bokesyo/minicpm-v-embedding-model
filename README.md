# OpenMatch V3

## 快速上手

### 模型训练

1. embedding模型在一个基座模型上训练，基座模型可以是Bert或Decoder only的，Bert取CLS token，Decoder only取最后一个非mask token。

2. 如果用openmatch训练模型，启动点在 `src/openmatch/driver/train_dr.py`，包括加载数据集和模型，继承了transformers.Trainer，有好处也有不方便。

3. 关于modeling，看 `src/openmatch/modeling/dense_retrieval_model.py` 的 `DRModel` 类，不需要的部分已经全部注释掉。这里分布式对比学习训练需要用到 `dist_gather_tensor` 方法，在每个GPU上收集batch内所有样本的encoded表征，这里的梯度是断的，但并不影响梯度反向传播的正确性。（需要思考3小时左右，可跳过）。损失函数是 CrossEntropyLoss，需要首先gather所有gpu上的reps，然后计算dot product，然后反向传播梯度，当然因为gather来的reps不带梯度，只在本gpu上计算的reps才有梯度，backward时只对这部分进行梯度反向传播，然后通过DDP同步多gpu的梯度，这种方法和一张超大gpu上跑所有的样本产生的梯度值成正比，只是需要再乘上world_size(gpu数)。有理论保证。

4. 关于trainer，看 `src/openmatch/trainer/dense_trainer.py`。

5. 关于数据，对比学习有一个query，多个corpus，corpus有正样本和负样本，我们这里1个query对应1个pos和1个neg。可用的数据总量在1000万左右。

6. 知乎集群训练，`openmatch/train.sh` 多机多卡训练。

7. 并行化优化策略：deepspeed和gradient_checkpointing，已经默认实现。这样可有效扩展上下文长度到1024，并实现高达1024的batch size。还没有集成的方法：gradCache，这是一种提前计算所有reps（不带梯度）然后gather，之后重新计算所有的reps，一次只算一小部分reps但带梯度，然后当场backward，用gradient accumulation实现计算正确的梯度。（需要理解gradcache和gradient accumulation和gradient_checkpointing的区别，他们很不一样）。deepspeed的作用是进一步降低模型参数和adam状态占用的显存，给大batchsize和长context腾出空间，与对比学习无关。


### 模型评估

1. 评测：在几个固定数据集上进行评测，入口在`eval.sh` -> `mteb_eval/evaluate_sbert_multi_gpu.py`，核心逻辑在 `/Library/beir/beir/retrieval/search/dense/exact_search_multi_gpu.py`，用了最小堆方法多卡并行评测。但似乎效率差。

2. openmatch自己实现的评测：`build_hn_step2.sh`，优点：大规模query和corpus，效率高，缺点：bug多，多卡并行有bug。内存管理有bug。

