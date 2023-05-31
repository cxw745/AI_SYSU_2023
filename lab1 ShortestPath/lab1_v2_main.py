import Romania_Path

if __name__ == '__main__':
    print('Welcome to inquire the shortest path!')
    while True:
        # 输入查询的两个城市，支持首字母和全称，不分大小写
        city_one = input('Enter the first city:')
        city_two = input('Enter the second city:')
        # 创建Romania类，使用其中的输入函数format_input输入两个城市，最终会输出结果并且写入日记
        Romania = Romania_Path.Romania()
        Romania.format_input(city_one, city_two)
        # 完成一轮查询后，询问是否要除楚日记和继续
        c = input('Do you want to clear diary? Y/N\n')
        if c == 'Y':
            Romania.clear_diary()
            print('Diary has been cleared!')
        e = input('Do you want to continue inquiry? Y/N\n')
        if e == 'Y':
            continue
        elif e == 'N':
            print('Welcome your next usage!')
            break
        else:
            print('Invalid enter!Continue.')
            continue
