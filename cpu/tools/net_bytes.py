#encoding=utf-8
import subprocess

def get_lo_bytes_now():
    try:
        result = subprocess.getoutput(['ifconfig lo |grep packets'])
        result = result.split('\n')[0]
        result = result.split(' ')[-3]
        result = int(result)
    except Exception as e:
        print(e)
        result = 0
    return result


if __name__ == '__main__':

    data = get_lo_bytes_now()
    print(data)
