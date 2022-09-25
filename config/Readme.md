# train_config.yaml
config 폴더의 train_config.yaml 부분입니다.

#### data
* train_make_select : boolean, True or False
  * 전처리 된 train 파일을 만드는 option입니다. 학습 중간에 결과가 끊겼을 때 다시 전처리를 수행하지 않도록 하는 option입니다.
  * True로 하면 전처리를 수행합니다.
* val_make_select : boolean, True or False
  * train_make_select를 valid data에 대해 수행하는 것입니다.
* train_normal_probability : float, 0 ~ 1
  * train의 batch에서 데이터를 뽑아낼 때마다 매번 0~1 사이의 값을 추출해 train_normal_probability percent보다 높으면 원래 이미지를 회전된 데이터로 만듭니다.
* val_normal_probability : float, 0 ~ 1
  * train_normal_probability의 valid 버전입니다. 다만 valid의 경우 한 번 만들면 더 이상 데이터가 바뀌지 않고 학습을 진행합니다.
* num_points : int
  * 각 이미지에서 사용하는 point 개수입니다.
* num_workers : int
  * dataloader에서 사용하는 num_worker 개수입니다.
* sampling_method : str, normal or random
  * normal은 점 간의 간격 등을 고려하여 num_points 개수를 선택하는 방법입니다. random보다 효과가 없어 random으로 사용하시면 될 것 같습니다. random : random으로 num_points 개수만큼 점을 추출합니다.
* model_sampling
  * str : boolean, True or False
    * PCT 코드에서 사용한 sampling 방법으로 True를 해야 작동합니다.
  * translate : boolean, True or False
    * PCT 코드에서 사용한 translate 방법입니다.
  * dropout : boolean, True or False
    * PCT 코드에서 사용한 방법으로 데이터의 point 중 일부를 쓰지 않는 방법입니다.
  * shuffle : boolean, True or False
    * PCT 코드에서 사용한 방법으로 point의 순서를 섞습니다.
* voxelize
  * voxel_downsampling : boolean, True or False
    * voxel_downsampling 사용여부 option입니다. True면 사용합니다.
  * voxel_size : float
    * voxel_downsampling에서 사용하는 voxel_size입니다.
* outlier(참고페이지 : http://www.open3d.org/docs/latest/tutorial/Advanced/pointcloud_outlier_removal.html)
  * outlier_remover : boolean, True or False
    * point의 outlier을 제거할지 설정합니다. True로 하면 삭제합니다.
  * nb_points : int
  * radius : float
* weight_method : str, list or random
  * list : data.py에서 미리 정해진 값을 기준으로 데이터를 회전시켜 augmentation을 수행합니다.
  * random : random으로 뽑힌 수치 값을 기준으로 데이터를 회전시켜 augmentation을 수행합니다.

#### model
* cv : int
  * cross_validation 적용 개수입니다.
  * 참고로 여기서 설정한 값만큼 모델링이 되는 것이 아니라, data를 cv 개수만큼 나눈 뒤 cv_num 번호를 valid로 하여 모델을 만듭니다.
  * 즉, 5개 cross validation을 수행하고자 한다면 cv : 5를 하고 cv_num을 0,1,2,3,4로 하여 5번 수행해야 합니다.
* cv_num : int
  * cross_validatoin 중 valid로 선택할 번호입니다.
* dropout : float, 0 ~ 1
  * model에서 사용하는 dropout값입니다.
  * 향후 train parameter로 넣을 예정입니다.
* load_serial_number : int or boolean
  * cross_validation을 사용할 때 쓰는 파라미터입니다.
  * 예를 들어 cross-validation에서 첫 번째 모델을 만든 뒤 두 번째 모델을 만들고자 한다면 load_serial_number에 기존에 만든 결과의 serial_number를 넣으면 됩니다. 
  * cross-validation을 사용하지 않는다면 False로 값을 넣습니다.
  
#### train : 학습 관련 파라미터입니다.

#### other
* random_seed : random seed로 사용하는 번호입니다.
* bench_mark : torch에서 bench_mark 설정 option 값입니다.

# pred_config.yaml
#### path
* train_serial : int
  * 학습할 때 생성된 serial_number를 넣습니다.
    * result의 train 폴더를 참조하시면 됩니다.
  * pred_cv : 학습했을 때 정한 cv_num을 넣습니다.
#### data
* test_make_select : boolean, True or False
  * 위의 train_make_select와 유사합니다.
* num_workers : int
  * DataLoader에서 사용하는 parameter입니다.

#### predict
* batch_size : int

#### other
* random_seed : int
* train_val_predict : boolean, True or False
  * val 데이터에 대한 predict 값을 생성할 때 사용하는 변수입니다.
