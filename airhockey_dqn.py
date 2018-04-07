import sys
from environment import Env
import pylab
import pygame
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K
import tensorflow as tf


EPISODES = 50000000


# 카트폴 예제에서의 DQN 에이전트
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 100000
        self.epsilon_decay_step = (self.epsilon_start-self.epsilon_end)/self.exploration_steps
        #self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000


        # 리플레이 메모리, 최대 크기 2000
        self.memory = deque(maxlen=2000)
        self.no_op_steps = 20

        # 모델과 타깃 모델 생성
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.avg_q_max = 0

        # 타깃 모델 초기화
        self.update_target_model()

        # 텐서보드 설정
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            'summary/airhockey_dqn', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            self.model.load_weights("./save_model/airhockey_dqn_trained.h5")

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        # 현재 상태에 대한 모델의 큐함수
        # 다음 상태에 대한 타깃 모델의 큐함수
        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        # 벨만 최적 방정식을 이용한 업데이트 타깃
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        self.model.fit(states, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)

# 각 에피소드 당 학습 정보를 기록
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        #episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('step/Episode', episode_duration)
        #tf.summary.scalar('Average Loss/Episode', episode_avg_loss)

        summary_vars = [episode_total_reward, episode_avg_max_q,episode_duration] #, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

if __name__ == "__main__":
    # CartPole-v1 환경, 최대 타임스텝 수가 500
    env = Env()
    state_size = len(env.get_state())
    action_size = env.action_size

    # DQN 에이전트 생성
    agent = DQNAgent(state_size, action_size)

    scores, episodes, global_step = [], [], 0

    for e in range(EPISODES):
        print("---------------------------------------------------")
        done = False
        score, step = 0, 0

        # env 초기화
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        gameExit = False


        for _ in range(random.randint(1,agent.no_op_steps)):
            state, _, _ = env.step(8)

        state = np.reshape(state, [1, state_size])

        while not done and not gameExit:
            if agent.render:
                env.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    gameExit = True
                    sys.exit()

            global_step += 1
            step += 1

            # 현재 상태로 행동을 선택
            action = agent.get_action(state)
            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # 리플레이 메모리에 샘플 <s, a, r, s'> 저장
            agent.append_sample(state, action, reward, next_state, done)

            # 매 타임스텝마다 학습
            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            agent.avg_q_max += np.argmax(agent.model.predict(state)[0])
            score += reward
            state = next_state

            if done:
                # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                agent.update_target_model()

                # 각 에피소드 당 학습 정보를 기록
                if global_step > agent.train_start:
                    stats = [score, agent.avg_q_max / float(step), step]
                             #agent.avg_loss / float(step)]
                    for i in range(len(stats)):
                        agent.sess.run(agent.update_ops[i], feed_dict={
                            agent.summary_placeholders[i]: float(stats[i])})
                    summary_str = agent.sess.run(agent.summary_op)
                    agent.summary_writer.add_summary(summary_str, e + 1)

                agent.avg_q_max = 0

                # 에피소드마다 학습 결과 출력
                scores.append(score)
                episodes.append(e)
                #pylab.plot(episodes, scores, 'b')
                #pylab.savefig("./save_graph/cartpole_dqn.png")
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon)

                # 1000 에피소드마다 모델 저장
                if e % 1000 == 0:
                    agent.model.save_weights("./save_model/airhockey_dqn_trained.h5")
