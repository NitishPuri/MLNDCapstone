____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
input_1 (InputLayer)             (None, 128, 128, 3)   0                                            
____________________________________________________________________________________________________
conv2d_1 (Conv2D)                (None, 128, 128, 64)  1792        input_1[0][0]                    
____________________________________________________________________________________________________
batch_normalization_1 (BatchNorm (None, 128, 128, 64)  256         conv2d_1[0][0]                   
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 128, 128, 64)  0           batch_normalization_1[0][0]      
____________________________________________________________________________________________________
conv2d_2 (Conv2D)                (None, 128, 128, 64)  36928       activation_1[0][0]               
____________________________________________________________________________________________________
batch_normalization_2 (BatchNorm (None, 128, 128, 64)  256         conv2d_2[0][0]                   
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 128, 128, 64)  0           batch_normalization_2[0][0]      
____________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)   (None, 64, 64, 64)    0           activation_2[0][0]               
____________________________________________________________________________________________________
conv2d_3 (Conv2D)                (None, 64, 64, 128)   73856       max_pooling2d_1[0][0]            
____________________________________________________________________________________________________
batch_normalization_3 (BatchNorm (None, 64, 64, 128)   512         conv2d_3[0][0]                   
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 64, 64, 128)   0           batch_normalization_3[0][0]      
____________________________________________________________________________________________________
conv2d_4 (Conv2D)                (None, 64, 64, 128)   147584      activation_3[0][0]               
____________________________________________________________________________________________________
batch_normalization_4 (BatchNorm (None, 64, 64, 128)   512         conv2d_4[0][0]                   
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 64, 64, 128)   0           batch_normalization_4[0][0]      
____________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)   (None, 32, 32, 128)   0           activation_4[0][0]               
____________________________________________________________________________________________________
conv2d_5 (Conv2D)                (None, 32, 32, 256)   295168      max_pooling2d_2[0][0]            
____________________________________________________________________________________________________
batch_normalization_5 (BatchNorm (None, 32, 32, 256)   1024        conv2d_5[0][0]                   
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 32, 32, 256)   0           batch_normalization_5[0][0]      
____________________________________________________________________________________________________
conv2d_6 (Conv2D)                (None, 32, 32, 256)   590080      activation_5[0][0]               
____________________________________________________________________________________________________
batch_normalization_6 (BatchNorm (None, 32, 32, 256)   1024        conv2d_6[0][0]                   
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 32, 32, 256)   0           batch_normalization_6[0][0]      
____________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)   (None, 16, 16, 256)   0           activation_6[0][0]               
____________________________________________________________________________________________________
conv2d_7 (Conv2D)                (None, 16, 16, 512)   1180160     max_pooling2d_3[0][0]            
____________________________________________________________________________________________________
batch_normalization_7 (BatchNorm (None, 16, 16, 512)   2048        conv2d_7[0][0]                   
____________________________________________________________________________________________________
activation_7 (Activation)        (None, 16, 16, 512)   0           batch_normalization_7[0][0]      
____________________________________________________________________________________________________
conv2d_8 (Conv2D)                (None, 16, 16, 512)   2359808     activation_7[0][0]               
____________________________________________________________________________________________________
batch_normalization_8 (BatchNorm (None, 16, 16, 512)   2048        conv2d_8[0][0]                   
____________________________________________________________________________________________________
activation_8 (Activation)        (None, 16, 16, 512)   0           batch_normalization_8[0][0]      
____________________________________________________________________________________________________
up_sampling2d_1 (UpSampling2D)   (None, 32, 32, 512)   0           activation_8[0][0]               
____________________________________________________________________________________________________
concatenate_1 (Concatenate)      (None, 32, 32, 768)   0           activation_6[0][0]               
                                                                   up_sampling2d_1[0][0]            
____________________________________________________________________________________________________
conv2d_9 (Conv2D)                (None, 32, 32, 256)   1769728     concatenate_1[0][0]              
____________________________________________________________________________________________________
batch_normalization_9 (BatchNorm (None, 32, 32, 256)   1024        conv2d_9[0][0]                   
____________________________________________________________________________________________________
activation_9 (Activation)        (None, 32, 32, 256)   0           batch_normalization_9[0][0]      
____________________________________________________________________________________________________
conv2d_10 (Conv2D)               (None, 32, 32, 256)   590080      activation_9[0][0]               
____________________________________________________________________________________________________
batch_normalization_10 (BatchNor (None, 32, 32, 256)   1024        conv2d_10[0][0]                  
____________________________________________________________________________________________________
activation_10 (Activation)       (None, 32, 32, 256)   0           batch_normalization_10[0][0]     
____________________________________________________________________________________________________
conv2d_11 (Conv2D)               (None, 32, 32, 256)   590080      activation_10[0][0]              
____________________________________________________________________________________________________
batch_normalization_11 (BatchNor (None, 32, 32, 256)   1024        conv2d_11[0][0]                  
____________________________________________________________________________________________________
activation_11 (Activation)       (None, 32, 32, 256)   0           batch_normalization_11[0][0]     
____________________________________________________________________________________________________
up_sampling2d_2 (UpSampling2D)   (None, 64, 64, 256)   0           activation_11[0][0]              
____________________________________________________________________________________________________
concatenate_2 (Concatenate)      (None, 64, 64, 384)   0           activation_4[0][0]               
                                                                   up_sampling2d_2[0][0]            
____________________________________________________________________________________________________
conv2d_12 (Conv2D)               (None, 64, 64, 128)   442496      concatenate_2[0][0]              
____________________________________________________________________________________________________
batch_normalization_12 (BatchNor (None, 64, 64, 128)   512         conv2d_12[0][0]                  
____________________________________________________________________________________________________
activation_12 (Activation)       (None, 64, 64, 128)   0           batch_normalization_12[0][0]     
____________________________________________________________________________________________________
conv2d_13 (Conv2D)               (None, 64, 64, 128)   147584      activation_12[0][0]              
____________________________________________________________________________________________________
batch_normalization_13 (BatchNor (None, 64, 64, 128)   512         conv2d_13[0][0]                  
____________________________________________________________________________________________________
activation_13 (Activation)       (None, 64, 64, 128)   0           batch_normalization_13[0][0]     
____________________________________________________________________________________________________
conv2d_14 (Conv2D)               (None, 64, 64, 128)   147584      activation_13[0][0]              
____________________________________________________________________________________________________
batch_normalization_14 (BatchNor (None, 64, 64, 128)   512         conv2d_14[0][0]                  
____________________________________________________________________________________________________
activation_14 (Activation)       (None, 64, 64, 128)   0           batch_normalization_14[0][0]     
____________________________________________________________________________________________________
up_sampling2d_3 (UpSampling2D)   (None, 128, 128, 128) 0           activation_14[0][0]              
____________________________________________________________________________________________________
concatenate_3 (Concatenate)      (None, 128, 128, 192) 0           activation_2[0][0]               
                                                                   up_sampling2d_3[0][0]            
____________________________________________________________________________________________________
conv2d_15 (Conv2D)               (None, 128, 128, 64)  110656      concatenate_3[0][0]              
____________________________________________________________________________________________________
batch_normalization_15 (BatchNor (None, 128, 128, 64)  256         conv2d_15[0][0]                  
____________________________________________________________________________________________________
activation_15 (Activation)       (None, 128, 128, 64)  0           batch_normalization_15[0][0]     
____________________________________________________________________________________________________
conv2d_16 (Conv2D)               (None, 128, 128, 64)  36928       activation_15[0][0]              
____________________________________________________________________________________________________
batch_normalization_16 (BatchNor (None, 128, 128, 64)  256         conv2d_16[0][0]                  
____________________________________________________________________________________________________
activation_16 (Activation)       (None, 128, 128, 64)  0           batch_normalization_16[0][0]     
____________________________________________________________________________________________________
conv2d_17 (Conv2D)               (None, 128, 128, 64)  36928       activation_16[0][0]              
____________________________________________________________________________________________________
batch_normalization_17 (BatchNor (None, 128, 128, 64)  256         conv2d_17[0][0]                  
____________________________________________________________________________________________________
activation_17 (Activation)       (None, 128, 128, 64)  0           batch_normalization_17[0][0]     
____________________________________________________________________________________________________
conv2d_18 (Conv2D)               (None, 128, 128, 1)   65          activation_17[0][0]              
====================================================================================================
Total params: 8,570,561
Trainable params: 8,564,033
Non-trainable params: 6,528
____________________________________________________________________________________________________