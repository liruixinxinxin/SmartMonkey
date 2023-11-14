def print_colorful_text(text, color):
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
    }
    end_color = '\033[0m'
    
    if color in colors:
        print(f"{colors[color]}{text}{end_color}")
    else:
        print(text)
        
        
def find_first_consecutive(lst, target):
    consecutive_count = 0

    for i, num in enumerate(lst):
        if num == target:
            consecutive_count += 1
            if consecutive_count == 1:
                first_position = i + 1  # 位置从1开始
            if consecutive_count == 4:
                return 1, first_position

        else:
            consecutive_count = 0

    return 0, None