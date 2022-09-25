## 아래 링크의 대회에서 썼던 코드입니다.
* https://dacon.io/competitions/official/235951/overview/description

## 아래 페이지의 방법론을 사용해서 만들었습니다.
* https://github.com/Strawberry-Eat-Mango/PCT_Pytorch
* Paper link: https://arxiv.org/pdf/2012.09688.pdf

## Setting : 아래 Requirements를 참고하시기 바랍니다.

### data
data 폴더에 아래 사이트에서 다운받은 train, test, submission 전부를 넣습니다.
* https://dacon.io/competitions/official/235951/data

### Requirements
* python : 3.8 
  * conda 
* 위의 github 코드에서는 python 3.7 이상의 버젼을 요구합니다.

```shell script
pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install pointnet2_ops_lib/.
```

## train 명령어
python main.py --execute=train

## test 명령어
python main.py --execute=test



