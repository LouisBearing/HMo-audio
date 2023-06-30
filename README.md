# Multi-scale [landmark-domain] Talking Head Generation

This page presents the results of our paper: "**A Comprehensive Multi-scale Approach for Speech and Dynamics Synchrony in Talking Head Generation**".
Below are qualitative comparisons of our results on 2D landmarks with prominent previous works, generated from VoxCeleb2 test set.

The code & model weights will be released soon.

## Qualitative comparison

Compared to previous methods, MS-Sync produces more natural head motion, and both lips and head motion are synced with the speech input thanks to a multi-scale synchrony loss.

**Sample #1**

https://github.com/LouisBearing/HMo-audio/assets/36541517/56d40da4-a917-4fd8-a746-fa460a1b8576

The Gif version:
![combined](https://github.com/LouisBearing/HMo-audio/assets/36541517/8a56949d-14ac-4642-a81d-507321744721)


**Sample #2**

https://github.com/LouisBearing/HMo-audio/assets/36541517/7348ea05-97a7-4180-83b1-885f1024d472

The Gif version:
![combined](https://github.com/LouisBearing/HMo-audio/assets/36541517/2d86f4c8-15d1-4ba8-b3fb-90c28561179b)


**Sample #3**



https://github.com/LouisBearing/HMo-audio/assets/36541517/daced343-2b84-4888-b9c1-08b6757e674c


The Gif version:
![combined](https://github.com/LouisBearing/HMo-audio/assets/36541517/40d05574-2947-40a4-ba40-ae7cef1319d4)


**Sample #4**

https://github.com/LouisBearing/HMo-audio/assets/36541517/93d6cf33-1c8a-4c6f-ba56-d72c65f4b0d1

The Gif version:
![combined](https://github.com/LouisBearing/HMo-audio/assets/36541517/3a76a502-6656-4b18-afa6-cf6bb2772982)


**Sample #5**

https://github.com/LouisBearing/HMo-audio/assets/36541517/f9277e7c-967c-47e7-b617-a3cbe74e5b33

The Gif version:
![combined](https://github.com/LouisBearing/HMo-audio/assets/36541517/fd9eff2e-a128-407c-8ec3-a41cc063c1d0)


Our model was trained on sequences of 40 frames and can produce outputs of much longer duration (120 frames in the examples above). However as it is an autoregressive generation process, error may accumulate if length exceeds a certain limit.


## Additional results

**Sample #1**

https://github.com/LouisBearing/HMo-audio/assets/36541517/fa68d9ef-d3b1-4e65-be6f-6b11c5aa202e


**Sample #2**

https://github.com/LouisBearing/HMo-audio/assets/36541517/32d52226-6660-46d9-a59f-bacdd33aa4fb


**Sample #3**

https://github.com/LouisBearing/HMo-audio/assets/36541517/fc2bef44-da7f-4179-b00b-216996b9f724





