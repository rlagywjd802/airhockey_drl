import pygame
import random


white = (255, 255, 255)
black = (0, 0, 0)
green = (0, 150, 0)
red = (255, 0, 0)
blue = (0, 0, 255)
yellow = (255, 255, 0)
light_blue = (147, 251, 253)
pink = (255, 100, 100)

HEIGHT = 300  # 픽셀 세로
WIDTH = 700  # 픽셀 가로

# goal
goalheight = 50
goalwidth = 10
goal1Position = [0, HEIGHT / 2 - 50]
goal2Position = [WIDTH - 10, HEIGHT / 2 - goalheight]
# paddle
paddle1Position = [WIDTH / 2 - 200, HEIGHT / 2]
paddleVelocity = 6
# puck
puckPosition = [WIDTH / 2, HEIGHT / 2]
puckVelocity = 6
score1 = 0
score2 = 0

# Initialize
pygame.init()
clock = pygame.time.Clock()


class Env():
    def __init__(self):
        self.action_space = ['u', 'd', 'r', 'l', 'ul', 'dl', 'ur', 'ul', 's']
        self.action_size = len(self.action_space)
        self.object = self.create_object()
        self.screen = self.build_screen()
        self.shapes = self.load_images()
        self.paddleDirection = [0, 0]
        self.puckDirection = [1, 1]
        self.serveDirection = -1
        self.steps_beyond_done = None
        self.count = 0

    def create_object(self):
        # Create Game Objects
        puck = pygame.Rect(puckPosition[0], puckPosition[1], 16, 16)
        paddle1 = pygame.Rect(paddle1Position[0], paddle1Position[1], 20, 20)
        return puck, paddle1

    def build_screen(self):
        # Board Size
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        return screen

    def load_images(self):
        puck = pygame.image.load('./images/disc.png')
        bluepad = pygame.image.load('./images/bluepad.png')
        redpad = pygame.image.load('./images/redpad.png')

        return puck, bluepad, redpad

    def set_info(self, state, index):
        temp = {}  # 딕셔너리 만들기
        temp['state'] = state
        temp['index'] = index

        self.info.append(temp)  # rewards 뒤에 temp 삽입

    def reset(self):
        pygame.display.update()
        self.paddleDirection = [0, 0]
        self.puckDirection = [self.serveDirection, self.serveDirection]

        # puck와 paddle 초기화
        self.object[0].x = WIDTH/2
        self.object[0].y = random.randint(50, HEIGHT-50)

        return self.get_state()

    def move_puck(self, target, paddle1):

        puck_x, puck_y = target.x, target.y

        puck_x += self.puckDirection[0] * puckVelocity
        puck_y += self.puckDirection[1] * puckVelocity

        if (puck_x <= target.width) and (puck_y <= HEIGHT / 2 + goalheight) and (puck_y >= HEIGHT / 2 - goalheight):
            self.serveDirection = -1
        elif (puck_x >= WIDTH - target.width - goalwidth) and (puck_y <= HEIGHT / 2 + goalheight) and (puck_y >= HEIGHT / 2 - goalheight):
            self.serveDirection = 1
        if puck_x - 8 < 0 or puck_x + 8 > WIDTH:
            self.puckDirection[0] *= -1
        if puck_y - 8 < 0 or puck_y + 8 > HEIGHT - target.height:
            self.puckDirection[1] *= -1

        if target.colliderect(paddle1):
            self.puckDirection[0] *= -1


        # object's position update
        self.object[0].x = puck_x
        self.object[0].y = puck_y

        # draw object
        pygame.draw.rect(self.screen, pink, self.object[0])
        # self.screen.blit(self.shapes[0], (puck_x - 6, puck_y - 6))
        s_ = [puck_x, puck_y]

        return s_

    def move_paddle(self, target, action):

        paddle1_x, paddle1_y = target.x, target.y

        if action == 0:  # ↑
            self.paddleDirection = [0, -1]
        elif action == 1:  # ↓
            self.paddleDirection = [0, 1]
        elif action == 2:  # →
            self.paddleDirection = [1, 0]
        elif action == 3:  # ←
            self.paddleDirection = [-1, 0]
        elif action == 4:  # ↖
            self.paddleDirection = [-1, -1]
        elif action == 5:  # ↘
            self.paddleDirection = [1, 1]
        elif action == 6:  # ↗
            self.paddleDirection = [1, -1]
        elif action == 7:  # ↙
            self.paddleDirection = [-1, 1]
        elif action == 8:  # .
            self.paddleDirection = [0, 0]

        paddle1_x += self.paddleDirection[0] * paddleVelocity
        paddle1_y += self.paddleDirection[1] * paddleVelocity

        if paddle1_y < 0:
            paddle1_y = 0
        elif paddle1_y > HEIGHT - 0.5 * target.width:
            paddle1_y = HEIGHT - 0.5 * target.width

        if paddle1_x < 0:
            paddle1_x = 0
        elif paddle1_x > abs((paddle1_y - 0.5 * HEIGHT) * (paddle1_y - 0.5 * HEIGHT) - (0.5 * HEIGHT - target.width) * (
                0.5 * HEIGHT - target.width)) ** 0.5:
            paddle1_x = abs((paddle1_y - 0.5 * HEIGHT) * (paddle1_y - 0.5 * HEIGHT) - (0.5 * HEIGHT - target.width) * (
            0.5 * HEIGHT - target.width)) ** 0.5

        # object's position update
        self.object[1].x = paddle1_x
        self.object[1].y = paddle1_y

        # draw object
        pygame.draw.rect(self.screen, pink, self.object[1])
        # self.screen.blit(self.shapes[1], (paddle1_x - 6, paddle1_y - 6))

    def check_if_reward(self, puck, paddle1):

        puck_x, puck_y = puck[0], puck[1]

        # check_list 딕셔너리
        check_list = dict()

        check_list['if_goal'] = False
        check_list['rewards'] = 0

        # goal 먹혔을 때
        if (puck_x <= self.object[0].width) and (puck_y <= HEIGHT / 2 + goalheight) and (puck_y >= HEIGHT / 2 - goalheight):
            check_list['rewards'] = -100
            check_list['if_goal'] = True
            print("***goal1")
            self.count = 0

        # goal 넣었을 때
        elif (puck_x >= WIDTH - self.object[0].width - goalwidth) and (puck_y <= HEIGHT / 2 + goalheight) and (puck_y >= HEIGHT / 2 - goalheight):
            check_list['rewards'] = 100
            check_list['if_goal'] = True
            print("***goal1")
            self.count = 0

        # 공 칠때
        elif puck.colliderect(paddle1):
            if(self.count == 0):
                check_list['rewards'] = 50
                check_list['if_goal'] = False
                print("***hit")
                print(puck_x, puck_y)
            self.count += 1

        # 상대편 골대 주변
        elif (puck_x >= WIDTH) and ((puck_y > HEIGHT / 2 + goalheight) or (puck_y < HEIGHT / 2 - goalheight)):
            check_list['rewards'] = -10
            check_list['if_goal'] = False
            print("***goal post")
            print(puck_x, puck_y)
            self.count = 0
        else:
            self.count = 0

        return check_list

    def get_state(self):

        # puck과 paddle 위치 불러오기
        paddle_location = self.object[1]
        puck_location = self.object[0]

        states = list()

        # paddle 절대위치
        states.append(paddle_location[0])
        states.append(paddle_location[1])
        # paddle 속도
        states.append(self.paddleDirection[0])
        states.append(self.paddleDirection[1])

        # puck 절대위치
        states.append(puck_location[0])
        states.append(puck_location[1])
        # puck 속도
        states.append(self.puckDirection[0])
        states.append(self.puckDirection[1])

        return states

    def render(self):
        # 게임 그래픽 업데이트
        goal1 = pygame.Rect(goal1Position[0], goal1Position[1], 10, 100)
        goal2 = pygame.Rect(goal2Position[0], goal2Position[1], 10, 100)

        divline1 = [WIDTH / 2, 0]
        divline2 = [WIDTH / 2, HEIGHT]

        pygame.draw.rect(self.screen, light_blue, goal1)
        pygame.draw.rect(self.screen, light_blue, goal2)
        pygame.draw.line(self.screen, white, divline1, divline2, 5)
        pygame.draw.circle(self.screen, white, [0, 150], 150, 2)

        clock.tick(100)
        pygame.display.update()

    def step(self, action):

        self.screen.fill(black)
        self.move_paddle(self.object[1], action)  # agent
        self.move_puck(self.object[0], self.object[1])

        self.render()

        check = self.check_if_reward(self.object[0], self.object[1])
        done = check['if_goal']
        reward = check['rewards']

        s_ = self.get_state()

        return s_, reward, done
