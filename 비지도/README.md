# CycleGAN 5-Fold 교차검증 실험 파이프라인

## 개요

본 저장소는 [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)를 기반으로,
별도의 소스/타겟 이미지 데이터셋을 활용해 **5-폴드 교차검증을 통한 CycleGAN 학습 및 결과 분석**을 자동화한 파이프라인이다.

본 파이프라인을 통해 데이터셋을 5-폴드로 분할하고, 각 fold별로 CycleGAN 학습 및 평가를 진행하고, fold별로 결과를 비교할 수 있도록 구성했다.


## 폴더 구조

```
0607_cycleGAN/
├── main_cv.py
├── split_cv.py
├── train_cyclegan.py
├── test_cyclegan.py
├── config.yaml
├── datasets/
│   └── vangogh/
│       └── fold_0/ ~ fold_4/
├── results/
│   └── cv_fold_0/ ~ cv_fold_4/
├── checkpoints/
│   └── cv_fold_0/ ~ cv_fold_4/
└── pytorch-CycleGAN-and-pix2pix/
    ├── train.py
    └── test.py
    └── ...
```


## 데이터 경로

- **소스 데이터(입력):**  
  `/home/nas/data/myeonggu/3-1/비지도_섭러/dataset/source_256`에 위치했다.
- **타겟 데이터(스타일):**  
  `/home/nas/data/myeonggu/3-1/비지도_섭러/dataset/target_256`에 위치했다.


## 주요 파일 설명

- `config.yaml`  
  실험 환경설정(폴드 수, 데이터 경로, 하이퍼파라미터, GPU 설정 등)을 담당한다.

- `split_cv.py`  
  소스/타겟 이미지를 5-폴드로 분할하여 학습/테스트용 폴더 구조를 생성한다.

- `train_cyclegan.py`  
  각 fold별로 CycleGAN 학습을 실행한다. (GPU 사용, visdom 비활성화)

- `test_cyclegan.py`  
  각 fold별로 테스트 이미지를 변환하고 결과를 저장한다.

- `main_cv.py`  
  위 과정을 순차적으로 한 번에 실행하는 메인 파이프라인이다.


## 사용법

1. **환경 준비**
   - Python 3.x, PyTorch, pyyaml 등 필수 패키지를 설치해야 한다.
   - CycleGAN 공식 코드 디렉토리(`pytorch-CycleGAN-and-pix2pix`)를 하위에 준비해야 한다.

2. **설정파일(config.yaml) 수정**
   - 데이터 및 하이퍼파라미터, GPU 설정 등을 환경에 맞게 수정해야 한다.

3. **실행**
   ```bash
   python3 main_cv.py
   ```
   또는 단계별로 실행할 수 있다.
   ```bash
   python3 split_cv.py
   python3 train_cyclegan.py
   python3 test_cyclegan.py
   ```

4. **결과 확인**
   - 변환 결과는 `results/cv_fold_*/test_latest/images/`에서 확인할 수 있다.
   - 학습된 모델 및 로그는 `checkpoints/cv_fold_*/`에 저장된다.


## 결과 해석

- 각 fold의 `results/cv_fold_X/test_latest/index.html` 파일을 웹브라우저에서 열면,
  한 쌍의 이미지별로 다음과 같은 6개의 이미지가 표시된다.

  | 이름      | 의미                                                        |
  |-----------|-----------------------------------------------------------|
  | real_A    | 실제 소스(입력) 이미지 (예: 일반 사진)                     |
  | fake_B    | real_A를 타겟(스타일) 도메인으로 변환한 결과               |
  | rec_A     | fake_B를 다시 소스 도메인으로 역변환(reconstruction)한 결과|
  | real_B    | 실제 타겟(스타일) 이미지 (예: 반 고흐 그림)                |
  | fake_A    | real_B를 소스 도메인으로 변환한 결과                       |
  | rec_B     | fake_A를 다시 타겟 도메인으로 역변환(reconstruction)한 결과|

- 이를 통해 CycleGAN의 스타일 변환 성능과 정보 보존(cycle consistency) 여부를 시각적으로 평가할 수 있다.
- 학습 손실값 변화는 `checkpoints/cv_fold_X/loss_log.txt`에서 확인할 수 있다.


## 참고/유의사항

- **Visdom 시각화 비활성화:**  
  visdom 관련 에러를 방지하기 위해 학습 시 `--display_id -1` 옵션을 사용했다.
- **여러 GPU 사용:**  
  `gpu_ids` 항목을 통해 원하는 GPU 번호를 지정할 수 있다.
- **정량적 평가:**  
  FID, IS 등 자동 평가 코드가 필요하면 별도 구현이 필요하다.


## 현재 상황 요약

1. CycleGAN 파인튜닝
원본(예: 현실 이미지) → 고흐 스타일 이미지 변환 학습을 시도.
하지만 실제로는 이미지의 형태 변환이 거의 일어나지 않았고, 기대한 스타일 변환이 부족.
https://github.com/brighteen/warehouse/tree/main/%EB%B9%84%EC%A7%80%EB%8F%84

2. DiT(Stable Diffusion 계열) 파인튜닝 시도
 더 강력한 모델(텍스트-이미지 생성 기반)을 적용하고자 시도함.
일부 이미지에 대해 프롬프트 작성.
환경 문제(CUDA, NCCL, PyTorch 버전 충돌)때매 막힘

3. 페어링 데이터 문제
원본 이미지와 변환된 고흐 스타일 이미지가 "직접적으로 1:1로 연결된 쌍(paired)이 없는 상황"임.
즉, **A(원본) ↔ B(고흐)**의 정합성이 없는 데이터셋이므로,
CycleGAN처럼 "비지도 도메인 변환" 모델만이 활용 가능하고,
Pix2Pix 같은 "지도학습 기반 1:1 매핑" 모델은 사용할 수 없음.

4. CycleGAN으로 생성한 사진도 페어링 데이터로 사용할 수 있지만 이미지 형태 변환이 안된 데이터라 파인튜닝해도 똑같이 형태변환은 안될거같음.