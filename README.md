# MSTAN
- the data is from [link](https://github.com/Bob05757/Renewable-energy-generation-input-feature-variables-analysis)
- first of all, I train a model  **MSTAN**, the paper is **Multi-Source and Temporal Attention Network for Probabilistic Wind Power Prediction**, but the result is not good at all..
- then we trained the Seq2seq mdoel, very simple but it works very well..
## Detail for Seq2seq model
- See the result at **show_data.ipynb**
- 15min a point, use **48** data to predict the next **48** data 
- the feature that we use is just **wind speed**, **wind angle** , The rest of the data is not very impactful, but adding them will improve performance a bit
- from this actual wind speed data, the result is quite good with a Mean Absolute Error about **3.2** for a turbine capacity of **100MW**
- 对于概率预测，我觉得直接去预测概率模型的分布，是是不合理而且效果很不好的。最一开始我是用MSTAN去预测一个Beta分布的概率，最终结果非常不好
- 一个比较好的概率预测可以这样的到：使用多个模型预测同样的结果得到一个序列[x1,x2,x3,x4...xn]，然后对这个序列进行KDE(kernel density estimation)分析，得到的概率结果可能才会更可信一些。

## picture of the result
![image](https://github.com/liuyalin-tanguy/MSTAN/assets/49784245/c4f3a15a-e31a-4949-a9ad-bcb34d6f3ad2)

