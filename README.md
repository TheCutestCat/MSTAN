# MSTAN
- the data is from [link](https://github.com/Bob05757/Renewable-energy-generation-input-feature-variables-analysis)
- first of all, I train a model  **MSTAN**, the paper is **Multi-Source and Temporal Attention Network for Probabilistic Wind Power Prediction**, but the result is not good at all..
- then we trained the Seq2seq mdoel, very simple but it works very well..
## Detail for Seq2seq model
- 15min a point, use **48** data to predict the next **48** data 
- the feature that we use is just **wind speed**, **wind angle** , The rest of the data is not very impactful, but adding them will improve performance a bit
- from this actual wind speed data, the result is quite good with a Mean Absolute Error about **3.2** for a turbine capacity of **100MW**

## picture of the result
![image](https://github.com/liuyalin-tanguy/MSTAN/assets/49784245/c4f3a15a-e31a-4949-a9ad-bcb34d6f3ad2)

