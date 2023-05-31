from Resolution_v1 import Resolution, format_input, S
import time

if __name__ == '__main__':
    # 做个菜单，读取文件或者自由输入
    num = 0
    while True:
        S.clear()
        choice_1 = input('请输入括号中的数字以选择：\n（1）自由输入\n（2）查看案例\n')
        if choice_1 == '1':
            print('请输入你的内容：')
            num = int(input())
            for i in range(num):
                clause = input()
                format_input(clause)
        elif choice_1 == '2':
            choice_2 = input('请输入括号中的数字选择你想查看的案例：\n（1）Aipine Club（2）Graduate Student（3）Block World\n')
            filename = ''
            if choice_2 == '1':
                filename = 'test1.txt'
            elif choice_2 == '2':
                filename = 'test2.txt'
            elif choice_2 == '3':
                filename = 'test3.txt'
            with open(filename, encoding='UTF-8') as f:
                num = int(f.readline().replace('\n', ''))
                print(num, end='\n')
                clauses = f.readlines()
                for clause in clauses:
                    print(clause, end='')
                    format_input(clause)
        start = time.time()
        Resolution(num)
        end = time.time()
        print('\nRunning time %.6f sec' % (end - start))
        choice_3 = input('\n继续使用/退出：y/n\n')
        if choice_3 == 'y':
            continue
        elif choice_3 == 'n':
            break
