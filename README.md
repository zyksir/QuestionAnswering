### 互联网数据挖掘大作业

##### 数据统计
```python3
from collections import defaultdict
q = defaultdict(lambda: [])
q2pos = defaultdict(lambda: 0)
q2neg = defaultdict(lambda: 0)
with open("train-set.data", "r") as f:
    for line in f:
        line = line.strip().split("\t")
        q[line[0]].append((line[1], line[2]))
        if line[2] == "0":
            q2neg[line[0]] += 1
        else:
            q2pos[line[0]] += 1
```
- train_dataset: 
    - 13114个问题，
    - 264415条数据，13580个是问题、正确答案，250836个是问题、错误答案
    - 一个问题可能会有{0～10}个正确答案，414个问题正例数目大于负例数目，4个问题没有正确答案
    - 一个问题可能会有{1～30}个错误答案，所有的问题均有错误答案
    
#### Dataloader
- 可能的办法:
    - 给loss加权重
    - self-adversarial negative sampling, 负例使用数据增强生成，因为负例句子的子集一定也没有答案
