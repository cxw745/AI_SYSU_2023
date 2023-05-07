# alpha-beta剪枝的中国象棋

#### 文件介绍

代码在`code`文件夹中，演示在`result`文件夹中。

**在code中：**

`images`是有些所需的图片。

`Test1.py`和`Test2.py`是main函数的程序，需要运行这个文件，主要功能是加载游戏资源和控制游戏进程的运行。

其他`.py`文件都是各自的类，具体功能如下：

> Chess.py 实现棋子类
> 
> ChessAI1.py 实现AI棋手类,实现alpha-beta搜索算法.
> 
> ChessAI2.py 在ChessAI1.py的技术上实现历史表优化.
> 
> Chessboard.py 实现棋局类,包括棋盘和棋子.
> 
> ClickBox.py 实现可点击棋子类.
> 
> Dot.py 实现当前棋子可以的走法类.
> 
> Game.py 实现裁判类,控制棋局进行.

#### 代码运行

代码实现基于python3.9.

需要安装`pygame`库。

```
pip install pygame
```

运行`Test1.py`程序，是无历史启发优化的Alpha-Beta。

运行`Test2.py`程序，是有历史启发优化的Alpha-Beta。
