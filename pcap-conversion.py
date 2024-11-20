import os


if __name__ == '__main__':
    PATH = r""
    file_list = os.listdir(PATH)
    for file in file_list:
        try:
            if not file.endswith('.pcapng'):
                continue
            file_path = PATH + '/' + file
            os.system('editcap -F libpcap {} {}'.format(file_path, file_path[:-2]))
            os.system('rm {}'.format(file_path))
        except Exception as e:
            print('error')    
    print('done')    