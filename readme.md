# sequential-recommendation

This repository contains implementations of various state-of-the-art algorithms for generating recommendations in a sequential, session-based process. The algorithms herein do not rely on standard recommender systems techniques such as user modelling, instead aiming to capture preferences from the isolated, sequential interactions exclusively. Due to this they are more robust to the cold-start problem and enable enhanced privacy-preservation, while being readily extensible to incorporate long-term user information. 

## Dataset Details

### Datasets

Currently, the repository supports two datasets:

#### Retail-Rocket

The data and a detailed description of the retail-rocket dataset can be found [here](https://www.kaggle.com/retailrocket/ecommerce-dataset).

For our interests, the dataset contains an 'events' file which is a '.csv' file containing the following columns: [timestamp, visitorid, event, itemid, transactionid] The 'visitorid' column corresponds to a session or user ID. The 'event' column contains information about the specific type of event, possibilities include click, add-to-cart and purchase events. To be consistent with the Recsys15 dataset we treat add-to-carts as purchases and drop purchase information from the dataset.


#### Recsys15: Yoochose

A detailed description of the dataset can be found in [6] (at the time of writing the data was no longer available for download).

The two files of interest for our purposes are yoochose-buys.dat and yoochose-clicks.dat, which we append to each other with the appropriate event information set. The data format requires no special processing (aside from some columns re-naming) to fit that of retail-rocket.

### Preprocessing

To make sure that all items and users (i.e., sessions) have a reasonable number of interactions, we remove all users and items that have fewer than three occurrences in the dataset.


## Models

This repository contains implementations of the below state-of-the-art models. Note that it is not intended as a reproduction study, and therefore not all functionality is implemented (e.g., user-specific modelling for GRU4Rec). The choice of models and some parameter settings were inspired by [7].

### GRU4Rec [1]

A gated recurrent unit (GRU) based neural network architecture for session-based recommendation. It is inspired by the ability of recurrent neural networks to model sequential data. The input embeddings are fed through recurrent GRU layers. The output of the last layer is used to compute interaction probabilities for the next item in the sequence.

### CASER [2]

A convolutional neural network architecture for session-based recommendation, inspired by traditional convolutional models for image recognition. It learns an embedding matrix together with the horizontal and vertical 2d-convolutional filters applied on it.

### NextItNet [3]

A convolutional neural network architecture for session-based recommendation, partially inspired by the CASER model. It employs stacked 1d-convolutional transformations on top of the embedded items.

### SAS4Rec [4]

Implements a self-attention based neural network for session-based recommendation. The attention mechanism is similar to that of the Transformer [5] architecture, encapsulated in stackable self-attention blocks with layer normalization, dropout and non-linear pointwise feed-forward transformations.

### Random

In addition to the above models, we implement a baseline that randomly chooses the next item to recommend to a user.



## Example Runs

All runs are carried out on an NVIDIA RTX2070 GPU. The learning rate is fixed at lr = 0.01. Results are reported using top_k = [5, 10, 15, 20].

### Retail-Rocket

#### CASER

| batch_size   | horizontal_filters | dropout | hidden_dim   | max_seq_len | epochs | Hit-Rates (click, buy) | NDCGs (click, buy) |
|--------------|-----------|------------|--------------|-----------|------------|-----------|------------|
| 256 |   [2, 3, 4]    | 0.25       |  64 | 10 | 20 | ([0.2324, 0.2688, 0.2901, 0.3057], [0.3422, 0.3810, 0.4033, 0.4185])| ([0.1871, 0.1989, 0.2045, 0.2082], [0.2857, 0.2982, 0.3041, 0.3076])|
| 256 |   [2, 3, 4]    | 0.25       |  128 | 10 | 20 | ([0.2322, 0.2682, 0.2878, 0.3013], [0.3426, 0.3847, 0.4052, 0.4175])| ([0.1857, 0.1974, 0.2026, 0.2058], [0.2855, 0.2992, 0.3047, 0.3076])|
| 256 |   [2, 3, 4]    | 0.5       |  64 | 10 | 5 | ([0.2297, 0.2675, 0.2886, 0.3035], [0.3654, 0.4065, 0.4312, 0.4484])| ([0.1859, 0.1981, 0.2037, 0.2072], [0.3062, 0.3195, 0.3261, 0.3301])|
| 256 |   [2, 3, 4, 5]    | 0.5       |  128 | 10 | 5 |([0.2591, 0.2979, 0.3189, 0.3349], [0.4149, 0.4563, 0.4760, 0.4919]) |([0.2099, 0.2224, 0.2280, 0.2318], [0.3499, 0.3633, 0.3685, 0.3723]) |
| 256 |   [2, 3, 4, 5]    | 0.5       |  256 | 10 | 5 |([0.2566, 0.2959, 0.3179, 0.3330], [0.4239, 0.4752, 0.4973, 0.5131])| ([0.2069, 0.2197, 0.2255, 0.2291], [0.3505, 0.3671, 0.3730, 0.3767])|



#### NextItNet

| batch_size   | dilated_channels | dilations| kernel_size | dropout | hidden_dim   | max_seq_len | epochs | Hit-Rates (click, buy) | NDCGs (click, buy) |
|--------------|-----------|------------|--------------|-----------|------------|-----------|------------|------------|------------|
| 256 |   64    |[1, 2, 4, 1, 2, 4]| 3 | 0.25       |  64 | 10 | 20 | ([0.1916, 0.2324, 0.2558, 0.2728], [0.4207, 0.4793, 0.5082, 0.5309])| ([0.1484, 0.1616, 0.1678, 0.1718], [0.3405, 0.3597, 0.3673, 0.3727])|
| 256 |   64    |[1, 2, 4, 1, 2, 4]| 3 | 0.25       |  128 | 10 | 20 | ([0.1667, 0.2018, 0.2232, 0.2390], [0.3718, 0.4261, 0.4544, 0.4735])| ([0.1294, 0.1408, 0.1465, 0.1502], [0.2974, 0.3151, 0.3225, 0.3271])|
| 256 |   128    |[1, 2, 4, 1, 2, 4]| 3 | 0.25       |  64 | 10 | 20 | ([0.1895, 0.2301, 0.2546, 0.2719], [0.4037, 0.4640, 0.4926, 0.5179])| ([0.1477, 0.1609, 0.1674, 0.1715], [0.3297, 0.3493, 0.3569, 0.3628])|
| 256 |   196    |[1, 2, 4, 1, 2, 4]| 3 | 0.25       |  64 | 10 | 20 | ([0.1922, 0.2309, 0.2542, 0.2710], [0.4140, 0.4767, 0.5059, 0.5281])| ([0.1504, 0.1630, 0.1691, 0.1731], [0.3342, 0.3546, 0.3623, 0.3676])|
| 256 |   64    |[1, 2, 4]| 3 | 0.25       |  128 | 10 | 20 | ([0.1600, 0.1956, 0.2173, 0.2329], [0.3611, 0.4233, 0.4554, 0.4750])| ([0.1236, 0.1351, 0.1408, 0.1445], [0.2855, 0.3057, 0.3141, 0.3188])|
| 256 |   64    |[1, 2, 4, 8, 1, 2, 4, 8]| 3 | 0.25       |  64 | 10 | 20 | ([0.1922, 0.2333, 0.2569, 0.2740], [0.4209, 0.4765, 0.5108, 0.5339])| ([0.1499, 0.1632, 0.1695, 0.1735], [0.3356, 0.3535, 0.3626, 0.3680])|
| 256 |   64    |[1, 2, 1, 2, 1, 2]| 3 | 0.25       |  64 | 10 | 20 |([0.1919, 0.2328, 0.2574, 0.2745], [0.4115, 0.4786, 0.5133, 0.5365]) | ([0.1486, 0.1619, 0.1684, 0.1724], [0.3328, 0.3545, 0.3637, 0.3692])|
| 256 |   512    |[1, 2, 4, 8, 1, 2, 4, 8]| 3 | 0.25      |  128 | 10 | 20 |([0.1693, 0.2046, 0.2262, 0.2423], [0.3596, 0.4166, 0.4475, 0.4711]) |([0.1323, 0.1437, 0.1494, 0.1532], [0.2854, 0.3038, 0.3120, 0.3176]) |
| 256 |   512    |[1, 2, 4, 8, 1, 2, 4, 8]| 3 | 0.1       |  64 | 10 | 20 | ([0.1920, 0.2339, 0.2579, 0.2742], [0.4153, 0.4752, 0.5043, 0.5247]) |([0.1496, 0.1631, 0.1695, 0.1734], [0.3359, 0.3554, 0.3631, 0.3679])|
| 256 |   512    |[1, 2, 4, 8, 1, 2, 4, 8]| 3 | 0.5       |  64 | 10 | 20 | ([0.1937, 0.2339, 0.2593, 0.2761], [0.4136, 0.4739, 0.5037, 0.5267]) | ([0.1501, 0.1634, 0.1701, 0.1741], [0.3314, 0.3510, 0.3589, 0.3643])|
| 256 |   256    |[1, 2, 4, 8, 1, 2, 4, 8]| 3 | 0.5        |  64 | 10 | 5 | ([0.2323, 0.2800, 0.3060, 0.3247], [0.5247, 0.5872, 0.6114, 0.6272])| ([0.1783, 0.1938, 0.2006, 0.2050], [0.4262, 0.4464, 0.4527, 0.4564])|
| 256 |   512    |[1, 2, 4, 8, 1, 2, 4, 8]| 3 | 0.5        |  64 | 10 | 5 | ([0.2283, 0.2758, 0.3025, 0.3209], [0.5209, 0.5794, 0.6082, 0.6243])| ([0.1760, 0.1913, 0.1984, 0.2028], [0.4242, 0.4432, 0.4509, 0.4547])|


#### Gru4Rec

| batch_size   | dropout |hidden_dim   | max_seq_len | epochs | Hit-Rates (click, buy) | NDCGs (click, buy) |
|-------------------------|------------|-----------|------------|------------|------------|------------|
| 256 |   0.25    | 64      | 10 | 20 | ([0.2168, 0.2587, 0.28416, 0.3013], [0.3312, 0.3797, 0.4050, 0.4183])|([0.1694, 0.1830, 0.1897, 0.1938], [0.2699, 0.2858, 0.2925, 0.2956])|
| 256 |   0.25    | 128       | 10 | 20|([0.1886, 0.2237, 0.24389, 0.2580], [0.2941, 0.3310, 0.3518, 0.3656])|([0.1477, 0.1591, 0.1644, 0.1678], [0.2414, 0.2533, 0.2588, 0.2621])|
| 256 |   0.5    | 64       | 10 | 5   |([0.1013, 0.1295, 0.14760, 0.1610], [0.1789, 0.2206, 0.2459, 0.2656])|([0.0770, 0.0861, 0.0909, 0.0941], [0.1405, 0.1542, 0.1609, 0.1655])|
| 256 |   0.5    | 128       | 10 | 5  |([0.2093, 0.2467, 0.26824, 0.2832], [0.3252, 0.3654, 0.3874, 0.4020])|([0.1650, 0.1771, 0.1828, 0.1864], [0.2684, 0.2815, 0.2873, 0.2908])|
| 256 |   0.5    | 256       | 10 | 5  |([0.1900, 0.2247, 0.24409, 0.2577], [0.2943, 0.3353, 0.3557, 0.3692])|([0.1493, 0.1606, 0.1657, 0.1690], [0.2399, 0.2532, 0.2586, 0.2618])|


#### SAS4Rec

| batch_size   | dropout | hidden_dim | num_sablocks | positionwise_feedforward_dim   | max_seq_len | epochs | Hit-Rates (click, buy) | NDCGs (click, buy) |
|--------------|-----------|------------|------------|-----------|------------|------------|------------|------------|
| 256 |   0.25    |64 | 1 | 64       | 10 | 20 |([0.2255, 0.2733, 0.3005, 0.3190], [0.5174, 0.5790, 0.6052, 0.6236]) |([0.1742, 0.1897, 0.1969, 0.2013], [0.4220, 0.4421, 0.4491, 0.4534])|
| 256 |   0.25    |64 | 2 | 64       | 10 | 20 |([0.2306, 0.2786, 0.3053, 0.3233], [0.5245, 0.5833, 0.6097, 0.6272]) |([0.1784, 0.1939, 0.2010, 0.2052], [0.4307, 0.4498, 0.4568, 0.4610]) |
| 256 |   0.25    |64 | 4 | 64       | 10 | 20 |([0.2335, 0.2813, 0.3080, 0.3262], [0.5048, 0.5646, 0.5914, 0.6109])| ([0.1798, 0.1953, 0.2024, 0.2067], [0.4159, 0.4354, 0.4426, 0.4472])|
| 256 |   0.25    |128 | 1 | 64       | 10 | 20|([0.1961, 0.2402, 0.2649, 0.2821], [0.4548, 0.5121, 0.5451, 0.5661])| ([0.1495, 0.1638, 0.1703, 0.1744], [0.3657, 0.3845, 0.3933, 0.3982])|
| 256 |   0.5    |64 | 1 | 64       | 10 | 5   |([0.2353, 0.2881, 0.3166, 0.3363], [0.5498, 0.6103, 0.6345, 0.6513])| ([0.1779, 0.1950, 0.2025, 0.2072], [0.4412, 0.4608, 0.4672, 0.4712])|


#### Random

| Hit-Rates (click, buy) | NDCGs (click, buy) |
|-----------|------------|
| ([7.944e-05, 0.0004, 0.0007, 0.001], [0.0, 0.0009, 0.001, 0.001])| ([4.9129e-05, 0.0001, 0.0002, 0.0003], [0.0, 0.0002, 0.0003, 0.0003])|

### Recsys15 - YOOchoose

To be done.

### CASER

### NextItNet

### Gru4Rec

### SAS4Rec

### Random

## References

[1] Hidasi, Balázs & Karatzoglou, Alexandros & Baltrunas, Linas & Tikk, Domonkos. 2015. Session-based Recommendations with Recurrent Neural Networks. 

[2] Jiaxi Tang and Ke Wang. 2018. Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding.

[3] Fajie Yuan, Alexandros Karatzoglou, Ioannis Arapakis, Joemon M. Jose, and Xiangnan He. 2019. A Simple Convolutional Generative Network for Next Item Recommendation.

[4] W. Kang and J. McAuley. 2018. Self-Attentive Sequential Recommendation.

[5] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need.

[6] David Ben-Shimon, Alexander Tsikinovsky, Michael Friedmann, Bracha Shapira, Lior Rokach, and Johannes Hoerle. 2015. RecSys Challenge 2015 and the YOOCHOOSE Dataset.

[7] Xin Xin, Alexandros Karatzoglou, Ioannis Arapakis, and Joemon M. Jose. 2020. Self-Supervised Reinforcement Learning for Recommender Systems.