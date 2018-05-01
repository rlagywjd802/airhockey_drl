# airhockey_drl
Development of Deep Reinforcement Learning algorithm to defeat a human player in Air Hockey Game.
This is the capstone project of Hyojeong Kim, Namki Yu and Kyungeun Kim in Hanyang University.

## v1.0
Using Deep Q Network

* hyper parameters
    - discount_factor = 0.99
    - learning_rate = 0.001
    - epsilon = 1.0
    - epsilon_decay_step = 9e-06
    - batch_size = 64
    - train_start = 1000

* neural network
    - 4 Layers
    - input size : 8    
        - paddle_position
        - paddle_velocity
        - puck_location
        - puck_velocity
    - output size : 9
        - action_space = ['u', 'd', 'r', 'l', 'ul', 'dl', 'ur', 'ul', 's']
    - hidden size : (24, 24)
    - initializer : 'he_uniform' 
    - loss : 'mse'
    - optimizer : 'Adam'

## v2.0
Goal : Using OpenAI Gym Environment
1. Add goal area
2. Add 'stop' action
3. Change CNN to DNN
