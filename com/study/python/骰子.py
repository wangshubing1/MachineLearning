#调用随机数
import random
class Game:
    def __init__(self,player1,player2):
        self.player1 = player1
        self.player2 = player2
        print("游戏初始化成功！")
    def startGame(self):
        self.player1.cast()
        self.player2.cast()
        print(self.player1)
        print(self.player2)
    def get_win(self):
        self.player1.guess_dice()
        self.player1.guess_dice()
        #判断谁赢
class Player:
    def __init__(self,name,sex,*dice):
        self.name = name
        self.sex = sex
        self.dices =dice#表示该玩家所拥有的骰子列表（元组）
    #抛色子
    def cast(self):
        for dice in self.dices:
            dice.move()
    def guess_dice(self):
        return (4,2)
    def __str__(self):
        play_dice_count_list = [self.dices[0].count, self.dices[1].count, self.dices[2].count]
        return "姓名为：%s的玩家抛掷的骰子列表为：%s"%(self.name,str(play_dice_count_list))
class Dice:
    def __init__(self):
        self.count = 0
    #骰子滚动的方法，应该返回该筛子的点数
    def move(self):
        self.count=random.randint(1,6)
d1=Dice()
d2=Dice()
d3=Dice()
d4=Dice()
d5=Dice()
d6=Dice()

p1=Player("player1","男",d1,d2,d3)
p2=Player("player2","男",d4,d5,d6)

for i in range(1,6):
    print("第%s次游戏情况=========>"%i)
    game =Game(p1,p2)
    game.startGame()


