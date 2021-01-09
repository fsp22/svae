def clear(log_file):
    with open(log_file, 'w') as f:
        f.write('')


def log(s, log_file):
    print(s)

    with open(log_file, 'a') as f:
        f.write(s + '\n')
