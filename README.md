### Channel Locality Block?

![Fig. 1. A Channel Locality block. We can simply see that it is a variant of a
Squeeze-and-Excitation block. Instead of using full connection layers to learn
the global channel correlation, we focus on the correlation between nearby
channels.](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2a1f2a3e-bd9b-424d-bd7b-98000f9eb672/Untitled.png)

Fig. 1. A Channel Locality block. We can simply see that it is a variant of a
Squeeze-and-Excitation block. Instead of using full connection layers to learn
the global channel correlation, we focus on the correlation between nearby
channels.

- squeeze-and-excitation network의 아이디어를 가지고 각 단계를 변화시킨 모델.
- global information 단계에선 Global MaxPooling, Global AveragePooling으로 각 채널 당 2개의 정보를 뽑아낸 후, convolution을 진행.
- Nearby Correlation 단계에선 SEnet과 다르게 convolution을 통해 local correlation을 반영.

1. Global Information
    - max pool, avg pool로 채널 특성을 반영
        
        ![Fig. 2. A diagram of the first step of a global information extraction part. We
        stack the global spatial information got by Global AveragePooling and Global
        Maxpooling.](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8cc63bd1-688e-45a0-ac31-c7de49070789/Untitled.png)
        
        Fig. 2. A diagram of the first step of a global information extraction part. We
        stack the global spatial information got by Global AveragePooling and Global
        Maxpooling.
        
    - 2x1 convolution으로 둘의 특성을 반영한 출력 생성
        
        ![Fig. 3. A diagram of the learning step of a global information extraction part.
        A group of 2 × 1 filters are applied to learn the relationship between two
        global information vectors.](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e5db0d5d-e64c-4eaa-a86b-13962cb9589b/Untitled.png)
        
        Fig. 3. A diagram of the learning step of a global information extraction part.
        A group of 2 × 1 filters are applied to learn the relationship between two
        global information vectors.
        
2. Nearby Correlation
    - convolution으로 인접 채널들의 값에서 중요한 값을 반영
        
        ![Fig. 4. A diagram of a nearby channel correlation extraction part. A single
        filter is applied to learn the correlation between the nearby channels.](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8b94408b-bc83-4633-91a5-45a7190842a2/Untitled.png)
        
        Fig. 4. A diagram of a nearby channel correlation extraction part. A single
        filter is applied to learn the correlation between the nearby channels.
        
- experiments
    
    ![TABLE V. EXPERIMENTS ON MODIFIED RESNET. (LEFT) PLANE CNN BENCHMARK.
    (MIDDLE) PLANE CNN WITH SE BLOCK. (RIGHT) PLANE CNN WITH
    C-LOCAL BLOCK.](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/911a9239-c70a-49a6-aea3-caf3a66358c5/Untitled.png)
    
    TABLE V. EXPERIMENTS ON MODIFIED RESNET. (LEFT) PLANE CNN BENCHMARK.
    (MIDDLE) PLANE CNN WITH SE BLOCK. (RIGHT) PLANE CNN WITH
    C-LOCAL BLOCK.
    
    ![TABLE VI. EXPERIMENTS ON MODIFIED RESNET. ACCURACY (%) THE CIFAR-10 TEST
    SET.](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e94e5ac1-1a00-463f-91f5-befe1d90830f/Untitled.png)
    
    TABLE VI. EXPERIMENTS ON MODIFIED RESNET. ACCURACY (%) THE CIFAR-10 TEST
    SET.
    
- 논문을 토대로 생성한 Channel Locality Block
    
    ```jsx
    import torch
    import torch.nn as nn
    
    class ChannelLocalityBlock(nn.Module):
        def __init__(self, C=16):
            super(ChannelLocalityBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=C//4, kernel_size=(2,1), stride=(1,1))
            self.conv2 = nn.Conv2d(in_channels=C//4, out_channels=1, kernel_size=(1,C//8), stride=(1,1))
            self.gap = nn.AdaptiveAvgPool2d((1,1))
            self.gmp = nn.AdaptiveMaxPool2d((1,1))
            self.valid = nn.AdaptiveAvgPool2d((1,C))
            self.ReLU = nn.ReLU(inplace=True)
            self.C = C
        def forward(self, input):
            x1 = self.gap(input) # torch.Size([B, C, 1, 1])
            x2 = self.gmp(input) # torch.Size([B, C, 1, 1])
            x = torch.cat((x1, x2), dim=2)
            x = x.view(-1, 1, 2, self.C) # torch.Size([B, 1, 2, C])
            x = self.conv1(x)
            x = self.ReLU(x)
            x = self.valid(x) # torch.Size([B, C // 4, 1, C])
            x = self.conv2(x)
            x = self.ReLU(x)
            x = self.valid(x) # torch.Size([B, 1, 1, C])
            x = x.view(-1, self.C, 1, 1)
            return x * input
    ```
