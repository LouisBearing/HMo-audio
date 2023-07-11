# Multi-scale [landmark-domain] Talking Head Generation

This page presents the results of our paper: "**A Comprehensive Multi-scale Approach for Speech and Dynamics Synchrony in Talking Head Generation**".
Below are qualitative comparisons of our results on 2D landmarks with prominent previous works, generated from VoxCeleb2 test set.

The code & model weights will be released soon.

## Qualitative comparison

Compared to previous methods, MS-Sync produces more natural head motion, and both lips and head motion are synced with the speech input thanks to a multi-scale synchrony loss.

**Sample #1 (don't forget to switch audio on)**

You can view in **full screen** and advance **frame by frame** to compare lips movements with Ground truth and Prajwal et al.'s state-of-the-art visual dubbing method :point_down:.

https://github.com/LouisBearing/HMo-audio/assets/36541517/56d40da4-a917-4fd8-a746-fa460a1b8576

We also provide the gif version to better **compare the realism of the dynamics** between different methods:
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



## Additional results

Here we present additional results. **Notice how head motion correlates with (the intensity of) the speech signal** :point_down:.

Our model was trained on **sequences of 40 frames** and can produce outputs of **much longer duration** (120 frames in all examples presented on this page). However as it is an autoregressive generation process, error may accumulate if length exceeds a certain limit. We therefore also present several **failure cases** at the end.

**Sample #1**

https://github.com/LouisBearing/HMo-audio/assets/36541517/f194b329-9b11-40c6-a199-65018ecd99f8

**Sample #2**

https://github.com/LouisBearing/HMo-audio/assets/36541517/32d52226-6660-46d9-a59f-bacdd33aa4fb

**Sample #3**

https://github.com/LouisBearing/HMo-audio/assets/36541517/0ddce381-8939-479f-b9cb-3a5b9882038c

**Sample #4**

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/da98a311-241c-42c8-976d-815d10df3a4c

**Sample #5**

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/4407b1bf-7453-4d17-84f3-68df0ba6adea

**Sample #6**

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/6decfdc1-51a8-4352-9efb-09907cd16a83

**Sample #7**

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/ba54b549-d558-4b0c-a2ac-dda5404164c9

**Sample #8**

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/8a13e6f9-73d1-4598-9f7f-e8b7cb89b0fb

**Sample #9**

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/b90dc70f-23c0-497c-9170-be98b7b5f3ec

**Sample #10**

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/3bc36684-fc6b-4658-8cb6-e7b05bbe87d1

**Sample #11**

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/b2631b09-68bf-413a-a308-e420272bb821

**Sample #12**

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/5d5e6999-b504-41a2-93cb-867230c95602

**Sample #13**

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/c8be43db-a628-4db6-8e90-302507118e79

**Sample #14**

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/be093312-d899-4381-93d1-dd9059bfdc1f

**Sample #15**

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/0f12a930-950c-4138-b796-0baf0ccb7b2c

**Sample #16**

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/7a77d567-f9c9-4b17-85d7-35fe03df8ac8

**Sample #17**

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/85dff47c-e7c3-461a-a805-5d380e6037b5

**Sample #18**

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/f9e5eb3a-1db2-4d40-92ed-8d21e787b7bb

**Sample #19**

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/cbdddb01-3adc-4047-bed8-43b962a14fef

**Sample #20**

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/2f20476d-119a-4648-a527-91edcdeb7f55

**Sample #21**

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/ed57ab1c-53fb-4db0-a184-bfa894137b31

**Sample #22**

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/b73e15d6-d01f-46b3-ab75-065ae7eb43c5

**Sample #23**

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/9df6f13a-7356-4676-9b2f-b5bc290c1051

**Sample #24**

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/b9dda2a3-6f19-4502-b8e6-4673f65e33fe

---

### Failure cases

**Case 1** - Some samples can present a from of dynamics **mode collapse**:

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/ca8c81e5-c180-4f91-9644-d6ff2b73a9cb

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/f02ff4e4-9800-4390-a3c8-fc624959f6bf


**Case 2** -Also as our model generates 2D coordinates without rigid constraints some frames can be **unrealistic**, even on sequence of medium length:

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/38f246d6-0fd4-495f-8880-e5ebe657b303

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/513856e5-6429-4224-8ebd-e1676e61b927

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/22755e55-0215-4a7e-86dd-e1b707523b73
