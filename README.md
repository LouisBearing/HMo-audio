# Multi-scale Talking Head Generation

This page presents the results of our paper: "**A Comprehensive Multi-scale Approach for Speech and Dynamics Synchrony in Talking Head Generation**".
Below are qualitative comparisons of our results with prominent previous works, selected randomly (i.e. not cherry picked) from the VoxCeleb2 test set.

The code & model weights will be released soon.

## Qualitative comparison

Compared to previous methods, MS-Sync produces more natural head motion, and both lips and head motion are synced with the speech input thanks to a multi-scale synchrony loss (don't forget to switch audio on).

https://github.com/user-attachments/assets/945a70a6-4945-49a2-ae9e-014006ef09d4

https://github.com/user-attachments/assets/f9da43c5-665a-45d2-bf3b-597add38e29b

https://github.com/user-attachments/assets/816f00a2-1db6-4ac6-aef8-2b5ba461ac4d

https://github.com/user-attachments/assets/314b15f6-e539-4767-b0a0-1d12b0d1b0b7

https://github.com/user-attachments/assets/b7f831a8-8247-4c10-bbad-705da5bb536e

https://github.com/user-attachments/assets/5618e8df-5ee9-436c-80b7-bb3956449b12

https://github.com/user-attachments/assets/c09717a9-033e-4d8d-89fd-b3e2f74aa827

https://github.com/user-attachments/assets/80df65a8-4fd0-499c-bce8-52ccb351d7cf

https://github.com/user-attachments/assets/6419aca6-9a5f-436b-affa-5f05c668eaa5

https://github.com/user-attachments/assets/6cca82cf-3106-43ec-ac20-f97fcd5d1ee1

https://github.com/user-attachments/assets/72fb6c4e-be42-4c1f-9d26-0c54cef2c8ef

https://github.com/user-attachments/assets/9008a447-9550-43f3-a766-a0017f18ed2d

https://github.com/user-attachments/assets/7827dd7a-2204-4f47-ac70-59dd4af3d9b9

https://github.com/user-attachments/assets/be181af4-2324-4f9e-807c-9689cc26c32c

https://github.com/user-attachments/assets/138657d4-4ca2-44d5-873e-fe14a40e9ab8

https://github.com/user-attachments/assets/769faa3c-bc48-4a49-a5dd-6cb540f366bb

https://github.com/user-attachments/assets/b4ffc35b-7e39-4b5c-b512-669103ce56d5

https://github.com/user-attachments/assets/5d68e961-a7ec-4b1e-bd7a-8b6228e34901





























### Failure cases

**Case 1** - Some samples can present a from of dynamics **mode collapse**:

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/ca8c81e5-c180-4f91-9644-d6ff2b73a9cb

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/f02ff4e4-9800-4390-a3c8-fc624959f6bf


**Case 2** - Also as our model generates 2D coordinates without rigid constraints some frames can be **unrealistic**, even on sequence of medium length:

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/38f246d6-0fd4-495f-8880-e5ebe657b303

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/513856e5-6429-4224-8ebd-e1676e61b927

https://github.com/SomeUser654321/Multi-scale-Talking-Head-Generation/assets/138246474/22755e55-0215-4a7e-86dd-e1b707523b73
