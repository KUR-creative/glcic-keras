# Globally and Locally Consistent Image Completion
Iizuka et al.의 17년도 image inapinting [논문](http://iizuka.cs.tsukuba.ac.jp/projects/completion/data/completion_sig2017.pdf)을 keras로 구현하였습니다. 

![szmc architecture](https://github.com/KUR-creative/SickZil-Machine/raw/master/doc/szmc-structure-eng.png)
image inpainting은 위 이미지에서 ComplNet을 담당하는 기술로, 이미지에서 지정한 부분을 자연스럽게 복원해 줍니다. 
만화 이미지에서 텍스트 영역을 "훼손된 부분"이라고 지정하고 inpainting 기술로 "복원"하면, 텍스트가 제거된 만화를 얻을 수 있습니다.

## 개발을 한 이유
ComplNet은 식질머신의 핵심 중의 핵심으로, 가장 먼저 개발하기 위해 노력한 모델입니다. 이 레포를 작성한 때는 
졸업 작품으로 [식질머신](https://github.com/KUR-creative/SickZil-Machine)을 만들던 18년도입니다.

그 당시에는 딥러닝 기반의 image inpainting 논문이 얼마 없었습니다. 몇몇 논문 중 가장 유망해 보이는 기술이 GLCIC였기에 이를 선택했습니다
그러나 당시에는 저자들이 [공개한 레포](https://github.com/satoshiiizuka/siggraph2017_inpainting)에 모델 학습 코드가 없었고, 
다른 사람들이 구현한 코드의 경우 메모리 leak으로 인해 학습이 중단되는 등 제대로된 학습 코드가 없었습니다.
그래서 직접 모델을 구현하고 학습을 시켰습니다.

## 결과
하지만 이 모델은 성능이 그리 좋지 못했으며, 매우 높은 컴퓨팅 자원을 필요로 했습니다. 
또한 학습 이후 전처리에 크게 의존해야 하는데, 흑백 만화에는 어울리지 않는 방식이었습니다.

## 결론
결론적으로 저는 [다른 모델](https://github.com/KUR-creative/old-cnet)을 써서 ComplNet을 구현하였습니다.
