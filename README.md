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

