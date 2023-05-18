# ModelingHelper
`PyTorch` 기반의 modeling의 기본 framework입니다.  

# How to
1. `sample` 폴더로 이동해서  
    ```python main.py --epochs 20 --patience 3 --dataset MNIST --gpu 3```  
    을 실행해보면 예시를 확인할 수 있습니다.
2. `main.py`의 `MyProject` 클래스를 이용해서 전체 흐름을 관리합니다.
3. 바꿔야할 부분은 다음과 같습니다.  
    1) `main.py`의 `main` 함수에서 dataset load를 진행하세요.  
        data는 `data` 폴더에 `dataset name` 하위폴더를 만들어서 관리하는 것을 권장합니다.
    3) `utils/utils.py`의 `parameter_parser` 함수 - 원하는 인자를 추가 변경하세요.
    4) `models/model.py`에 `MyProject`에서 선언할 최종 함수를 정의하세요.  
        기타 모듈은 새로운 파일에 정의하는 것을 권장합니다.  
        `__init__()`에 필요한 hyperparameter는 모두 argument parser로 관리하는 것을 권장합니다.
    4) `utils/metrics.py`에 적절한 metric 함수를 정의하세요.
    5) 기본적인 learning rate scheduler를 설정해두었습니다:  
        - 'main.py`의 `scheduler_dict`, `from torch.optim.lr_scheduler import ...`
        - `utils/utils.py`의 `possible_scheduler`, `set_scheduler_params`
4. 정상적으로 실행되면 `./results/{dataset_name}/%m-%d-%H-%M-%S/` 폴더 안에 다음과 같은 파일이 만들어 집니다:
    - `results.json`: train, validation loss와 validation, test metrics list
    - `loss_curve.png`: train, validation의 loss curve
    - `val_accuracy.png`: validation metric에 대한 curve
    - `model.pt`: best_valid_metric 갱신 시 저장되는 model state dict
    - `model_{epoch}.pt`: `args.safety_save` 설정 시 저장되는 model state dict

# LICENSE
This project is licensed under the MIT License. The license file can be found in LICENSE.
> I hope this repository is helpful to your coding
