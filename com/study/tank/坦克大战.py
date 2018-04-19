#coding:utf-8
import pygame,sys
from  pygame.locals import *
'''坦克大战的主窗口'''
class TankMain():
    #游戏开始
    def startGame(self):
        #创建一个屏幕的宽和高，和窗口特性(0，RESIZABLE ,FULLSCREEN),颜色位数
        screem=pygame.display.set_mode((640, 480),RESIZABLE ,32)
        #设置一个标题
        pygame.display.set_caption("坦克大战")
        while True :
            #设置背景屏幕为黑色
            screem.fill((0,0,0))
            #显示重置
            pygame.display.update()
    #游戏结束
    def stopGame(self):
        sys.exit()
        pass
    #设置抬头
    def set_title(self):
        pass

game=TankMain()
game.startGame()

