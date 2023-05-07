import sys
import pygame.display
from Game import *
from Dot import *
from ChessAI3 import *


def main():
    # 初始化pygame
    pygame.init()
    #名字
    pygame.display.set_caption('ChessAI3')
    # 播放音乐
    pygame.mixer.init()
    pygame.mixer.music.load('images\jay.flac')
    pygame.mixer.music.play(-1, 0)
    # 创建用来显示画面的对象（理解为相框）
    screen = pygame.display.set_mode((750, 667))
    # 游戏背景图片
    background_img = pygame.image.load("images/bg.jpg")
    # 游戏棋盘
    # chessboard_img = pygame.image.load("images/bg.png")
    # 创建棋盘对象
    chessboard = ChessBoard(screen)
    # 创建计时器
    clock = pygame.time.Clock()
    # 创建游戏对象（像当前走棋方、游戏是否结束等都封装到这个对象中）
    game = Game(screen, chessboard)
    game.back_button.add_history(chessboard.get_chessboard_str_map())
    # 创建AI对象
    ai_r = ChessAI(game.user_team)
    ai_b = ChessAI(game.computer_team)
    # 主循环
    while True:
        # AI行动
        if not game.show_win and game.AI_mode and game.get_player() == ai_b.team:
            time.sleep(0.5)
            # --------------------------------------------AI时间
            aitime1 = time.time()
            # --------------------------------------------AI时间

            # AI预测下一步
            cur_row, cur_col, nxt_row, nxt_col = ai_b.get_next_step(chessboard)
            # 选择棋子
            ClickBox(screen, cur_row, cur_col)
            # 下棋子
            chessboard.move_chess(nxt_row, nxt_col)
            # 清理「点击对象」
            ClickBox.clean()
            # 检测落子后，是否产生了"将军"功能
            if chessboard.judge_attack_general(game.get_player()):
                print("将军....")
                # 检测对方是否可以挽救棋局，如果能挽救，就显示"将军"，否则显示"胜利"
                if chessboard.judge_win(game.get_player()):
                    print("获胜...")
                    game.set_win(game.get_player())
                else:
                    # 如果攻击到对方，则标记显示"将军"效果
                    game.set_attack()
            else:
                if chessboard.judge_win(game.get_player()):
                    print("获胜...")
                    game.set_win(game.get_player())
            # --------------------------------------------AI时间
            aitime2 = time.time()
            print('AI_b time %.3f'%(aitime2-aitime1))
            # --------------------------------------------AI时间
            # 落子之后，交换走棋方
            game.back_button.add_history(chessboard.get_chessboard_str_map())
            game.exchange()
        else:
            time.sleep(0.5)
            # --------------------------------------------AI时间
            aitime1 = time.time()
            # --------------------------------------------AI时间

            # AI预测下一步
            cur_row, cur_col, nxt_row, nxt_col = ai_r.get_next_step(chessboard)
            # 选择棋子
            ClickBox(screen, cur_row, cur_col)
            # 下棋子
            chessboard.move_chess(nxt_row, nxt_col)
            # 清理「点击对象」
            ClickBox.clean()
            # 检测落子后，是否产生了"将军"功能
            if chessboard.judge_attack_general(game.get_player()):
                print("将军....")
                # 检测对方是否可以挽救棋局，如果能挽救，就显示"将军"，否则显示"胜利"
                if chessboard.judge_win(game.get_player()):
                    print("获胜...")
                    game.set_win(game.get_player())
                else:
                    # 如果攻击到对方，则标记显示"将军"效果
                    game.set_attack()
            else:
                if chessboard.judge_win(game.get_player()):
                    print("获胜...")
                    game.set_win(game.get_player())
            # --------------------------------------------AI时间
            aitime2 = time.time()
            print('AI_r time %.3f'%(aitime2-aitime1))
            # --------------------------------------------AI时间
            # 落子之后，交换走棋方
            game.back_button.add_history(chessboard.get_chessboard_str_map())
            game.exchange()

        # 显示游戏背景
        screen.blit(background_img, (0, 0))
        screen.blit(background_img, (0, 270))
        screen.blit(background_img, (0, 540))

        # 显示棋盘以及棋子
        chessboard.show_chessboard_and_chess()

        # 标记点击的棋子
        ClickBox.show()

        # 显示可以落子的位置图片
        Dot.show_all()

        # 显示游戏相关信息
        game.show()

        # 显示screen这个相框的内容（此时在这个相框中的内容像照片、文字等会显示出来）
        pygame.display.update()

        # FPS（每秒钟显示画面的次数）
        clock.tick(60)  # 通过一定的延时，实现1秒钟能够循环60次


if __name__ == '__main__':
    main()
