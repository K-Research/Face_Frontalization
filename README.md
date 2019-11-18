Result

Autoencoder :

![Autoencoder 1](https://user-images.githubusercontent.com/52855867/69034884-977f6e00-0a25-11ea-86d7-e6e067f1cbfb.png)

Deep Convolutional Generative Adversarial Network :
![DCGAN 1](https://user-images.githubusercontent.com/52855867/69035055-03fa6d00-0a26-11ea-8c38-2b434ed3176c.png)

![DCGAN 2](https://user-images.githubusercontent.com/52855867/69035092-1aa0c400-0a26-11ea-9d2b-804e59b1e4bd.png)
- 성별 구분이 어렵다.
- 같은 인물에 대해 예측 값의 편차가 크다.

Deep Convolutional Generative Adversarial Network + Super Resuolution Generative Adversarial Network :
![DCGAN + SRGAN 1](https://user-images.githubusercontent.com/52855867/69035169-55a2f780-0a26-11ea-987c-33080f14b165.png)
- 성별 구분이 가능하다.
- 이목구비가 더 선명하다.

- 개인에 대한 특성을 찾지 못한다.

Autoencoer (VGGFACE (VGG16)) :
![Autoencoder (VGG16) 1](https://user-images.githubusercontent.com/52855867/69035312-b4687100-0a26-11ea-8ca1-694d34fb8a85.png)
- 사람의 특성을 찾는다.

- 개인에 대한 특성을 찾지 못한다.

Deep Convolutional Generative Adversarial Network (VGGFACE (VGG16)):
![DCGAN (VGG16) 1](https://user-images.githubusercontent.com/52855867/69035397-eb3e8700-0a26-11ea-8e8f-75c3cdf3ce39.png)
- 사람에 대한 특성 추출이 개선되었다.
- 이미지의 선명도가 개선되었다.

- 좌, 우 얼굴이 다르다.
- 훈련 데이터에 없는 얼굴은 부정확하다.

Deep Convolutional Generative Adversarial Network (VGGFACE (VGG16)) + Super Resuolution Generative Adversarial Network (VGGFACE (RESNET50):
![DCGAN (VGG16) + SRGAN (RESNET50) 1](https://user-images.githubusercontent.com/52855867/69035412-f7c2df80-0a26-11ea-99b8-8611c43bcc5c.png)

- 사람에 대한 특성 추출이 개선되었다.
- 이미지의 선명도가 개선되었다.
- 좌, 우 얼굴이 동일하다.

- 그림처럼 나온다.
- 훈련 데이터에 없는 얼굴은 부정확하다.
