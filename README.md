## TensorFlow 기초
- tensorflow는 `tf`로 줄여 사용한다
    ```python
    import tensorflow as tf
    
    t = tf.constant(3)
    print(t)
    ```
- 자료형 tensor는 행렬(`matrix`)로 가중치(`w`) 연산이 잦은 딥러닝 연산에 적합하다
- tensor는 `shape`을 가지며 행렬의 크기와 차원을 다룬다.   `t.shape`을 확인가능하다
- tensor의 `datatype`은 크게 `정수(int)`와 `실수(float)`가 있다.
- keras를 tensor API로 딥러닝 프로토타입을 만들기 수월하게 한다. `tensor.keras`로 호출한다.
### Tensor 선언하기
  - `tf.constant`
    - 특정 값 하나를 선언할때
        ```python
        tf.constant(3)
        ```
    - 리스트를 선언할때
        ```python
        tf.constant([3,4,5])
        ```
    - 행렬을 선언할때
        ```python
        tf.constant([[1,2], [3,4]])
        ```
  - `tf.zeros`
    - 빈 리스트 만들기
        ```python
        tf.zeros(5)
        ```
    - 빈 행렬 만들기
        ```python
        tf.zeros([2,2]) # 2x2 행렬
        ```
    - 다차원 행렬 만들기
        ```python
        tf.zeros([2,2,3]) # 3x2x2 행렬
        ```
  - `tf.Variable`
    - 가중치(`w`)처럼 빈번하게 값이 변하는 변수에 적합함
    - Variable 값을 수정하려면 `t.assign()`을 한다
    - Variable의 값을 얻으려면 `t.numpy()`를 부른다
        ```python
        w = tf.Variable(1.0)
        w.assign(0.5)
        print(w.numpy()) # 값: 0.5
        ```
### Tensor 연산하기
  - 직접 연산자를 사용하거나 tensor 메서드를 사용한다
  - `덧셈`: `tf.add(a, b)`
  - `뺄셈`: `tf.subtract(a, b)`
  - `곱셈`: `tf.divide(a, b)`
  - `나눗셈`: `tf.multiply(a, b)`
  - `행렬곱`: `tf.matmul(a, b)`
### Tensor 자료형(dtype) 알아보기
  - tensor의 `datatype`은 크게 `정수(int)`와 `실수(float)`가 있다.
  - Tensor 자료형을 `float`로 강제하기
    - 실수를 한 개 이상 포함하기
        ```python
        tf.constant( [3.0, 4, 5])
        ```
    - `tf.float32` 명시하기
        ```python
        tf.constant([3,4,5], tf.float32)
        ```
    -`tf.cast` 메서드 사용하기

## 선형회귀(Linear Regression) 문제를 통해 딥러닝 과정 구현하기
1. 모델(`model`)과 레이어(`layer`) 생성하기
    - 모델을 작성하는 방식에 따라 `Sequential`, `Functional`, `Subclassing`으로 나뉜다
        - `Sequential`: layer를 순차적으로 add한다
            ```python
            from tensorflow.keras import models, layers

            # 기본
            model = models.Sequential()
            model.add(layers.Dense(64, activation='tanh'))
            model.add(layers.Dense(128, actionvation='tanh'))
            model.add(layers.Dense(1, activation='sigmoid'))
            # 단일 list로 한번에
            model = models.Sequential([
                layers.Dense(64, activation='tanh'),
                layers.Dense(128, activation='tanh'),
                layers.Dense(1, activation='sigmoid'),
            ])
            ```
        - `Functional`
        - `Subclassing`
    - 레이어는 보통 `Dense`를 많이 사용한다
    - 활성함수(`activation`)은 달성하고자 하는 목적에 따라 적절한 경우를 택한다
        - `sigmoid`
        - `tanh`
    - 마지막 레이어는 출력층으로 노드(`node`)를 1개로 정한다
2. 손실함수(`loss`)와 최적화방식(`optimizer`) 정하기(`compile`)
    - 손실함수(`loss function`): 예상값과 실제결과의 오차를 어떤 방식으로 처리하는지 결정함
        - `mse`(절대평균오차, mean square error)
        - `binary_crossentropy`: 이진분류 상황에 자주 사용된다
    - 최적화방식(`optimizer`): 학습률(`learning rate`)을 상황에 따라 적절하게 조정해준다
        - `adam`
    - 이진분류 상황 예시
        ```python
        model.compile(optimizer='adam', loss-'binary_crossentropy', metrics=['accuracy'])
        ```
        - `metrics`는 훈련과정에 수집할 정보를 정하는 것으로 여기서는 정확도(`accuracy`)다
3. 데이터셋(`dataset`) 준비하기
    - numpy array로 된 변인(`x`)과 결과(`y`)를 준비하는 과정으로 관례적으로 `train_x`와 `train_y` 변수에 담는다
    - 데이터 처리를 수월하게 하기 위해 `pandas`를 이용한다
    - 데이터셋 준비 예시
        ```python
        import pandas as pd

        df = pd.read_csv('XX.csv')
        print(df)
        '''
            admit    gre   gpa  rank
        0        0  380.0  3.21     3
        1        1  660.0  3.67     3
        2        1  800.0  4.00     1
        3        1  640.0  3.19     4
        4        0  520.0  2.33     4
        ..     ...    ...   ...   ...
        411      1  680.0  3.78     3
        412      0  650.0  3.11     1
        413      1  760.0  3.90     1
        414      0  540.0  3.44     2
        415      1  660.0  3.76     3
        '''
        df = df.dropna()

        train_x = df.loc(:, 'gre':'rank').values
        train_y = df['admit'].values
        ```
    - `df.dropna()`는 데이터 전처리 과정으로 누락된 불량 데이터를 삭제한다
    - `df.values`는 pandas의 `dataframe`을 `numpy array`로 변환하는 메서드다
4. 모델 여러차례 훈련(`fit`)하기
    ```python
    model.fit(train_x, train_y, epochs=1000)
    ```
    - X, y는 데이터셋을 말하며 numpy array 형태여야 한다
        - 만약 기본 python array라면 `np.array()`로 형식을 통일한다
    - `eopchs`는 전체 데이터 순회 횟수다
5. 학습한 모델을 근거로 문제 결과를 예상(`predict`)하기
    ```python
    result = model.predict(forcast_x)
    print(result)
    ```
