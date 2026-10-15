import os
import subprocess
import time
from pynvml.smi import nvidia_smi
# from pynvml_utils import nvidia_smi
import argparse
import json
import datetime
import requests
from dateutil.relativedelta import relativedelta
import sys, tty, termios
import threading
import socket
import zipfile

def initialize(arg):
    config = getJson('config')
    N_GPUS = config['N_GPUS']
    
    nowQ = {}
    for i in range(N_GPUS):
        nowQ[i] = None
    empty = {'done':[], 'now':nowQ, 'queue':[]}
    putJson(empty, 'Q')
    
    while True:
        Q = getJson('Q')
        free_gpu = getFreeGPU()
        if len(free_gpu) > 0:
            for idx in range(N_GPUS):
                job = Q['now'][str(idx)]
                if idx in free_gpu and job is not None:
                    now = datetime.datetime.now() + relativedelta(hours=9)
                    now_time = now.strftime('%b-%d %H:%M:%S')
                    end_job = {'exp':job['exp'], 'status':'Finished', 's_time':job['s_time'], 
                            'e_time':now_time, 'cmd':job['cmd']}
                    Q['now'][str(idx)] = None
                    Q['done'].append(end_job)
                    message = {"content": f"Your experiment {job['exp']} is now Finished! Please check the train process!"}
                    requests.post(config['discord_url'], data=message)
                    
            for idx in free_gpu:
                for i, Qjob in enumerate(Q['queue']):
                    now = datetime.datetime.now() + relativedelta(hours=9)
                    now = datetime.datetime.strptime(now.strftime('%b-%d %H:%M:%S'), '%b-%d %H:%M:%S')
                    if datetime.datetime.strptime(Qjob['Qtime'], '%b-%d %H:%M:%S') < now:
                        job = Q['queue'].pop(i)
                        log_dir = config['log_dir']
                        os.makedirs(f"{log_dir}/{job['exp']}", exist_ok=True)
                        os.system(f"CUDA_VISIBLE_DEVICES={idx} nohup {job['cmd']} 1>{log_dir}/{job['exp']}/nohup.out 2>&1 &")
                        # os.system(f"nohup {job['cmd']} 1>{log_dir}/{job['exp']}/nohup.out 2>&1 &")
                        now_time = now.strftime('%b-%d %H:%M:%S')
                        new_job = {'exp':job['exp'], 's_time':now_time, 'cmd':job['cmd']}
                        
                        Q['now'][str(idx)] = new_job
                        # # TODO
                        # for j in range(N_GPUS):
                        #     Q['now'][str(j)] = new_job
                        break
        Q = putJson(Q, 'Q')
        time.sleep(config['refresh_time'])

def init(arg):
    os.makedirs('./.dh', exist_ok=True)
    if not os.path.isfile('.dh/config.json'):
        defaultConfig()
    if isInitted() == 0:
        os.system('nohup python scheduler.py initialize 1>.dh/dh_log.out 2>&1 &')
        print("Successfully initialized the scheduler!")
    else:
        print("The scheduler is already working!")
    
def exit(arg):
    pid = isInitted()
    if pid == 0:
        print("The scheduler is not initialized!")
    else:
        os.system(f'kill {pid}')
        print("Successfully exited the scheduler!")

def push(args):
    exp, hours, cmd = args
    Q = getJson('Q')
    Qtime = datetime.datetime.now() + relativedelta(hours=9+hours)
    Qtime_txt = Qtime.strftime('%b-%d %H:%M:%S')
    job = {'Qtime':Qtime_txt,'exp':exp, 'cmd':cmd}
    Q['queue'].append(job)
    putJson(Q, 'Q')

def pop(pop_num):
    Q = getJson('Q')
    if pop_num == 0:
        pop_num = len(Q['queue']) - 1
    if pop_num == 'all':
        Q['queue'] = []
        job = None
    else:
        job = Q['queue'].pop(pop_num-1)
    putJson(Q, 'Q')
    return job

def move(arg):
    dep, dest = arg
    Q = getJson('Q')
    job = Q['queue'].pop(dep-1)
    Q['queue'].insert(dest-1, job)
    putJson(Q, 'Q')

def clear(idx):
    Q = getJson('Q')
    if idx == 0:
        Q['done'] = []
    else:
        Q['done'].pop(idx-1)
    putJson(Q, 'Q')

def kill(idx):
    Q = getJson('Q')
    job = Q['now'][str(idx)]
    cmd = job['cmd']
    out = subprocess.run(f'ps -ef | grep "{cmd[:-10]}"', shell=True, capture_output=True)
    out = str(out.stdout, 'utf-8').strip().split('\n')
    for proc in out:
        if 'grep' not in proc:
            pid = int(proc.split()[1])
            os.system(f'kill {pid}')
            break
    now = datetime.datetime.now() + relativedelta(hours=9)
    now_time = now.strftime('%b-%d %H:%M:%S')
    end_job = {'exp':job['exp'], 'status':'Killed', 's_time':job['s_time'], 
               'e_time':now_time, 'cmd':job['cmd']}
    Q['now'][str(idx)] = None
    Q['done'].append(end_job)
    putJson(Q, 'Q')
    
def see(arg):
    def quitChecker(q_list):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            # tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        q_list.append(ch)
    q_list = []
    ender = threading.Thread(target=quitChecker, args=(q_list,))
    ender.start()
        
    j = 0
    os.system('clear')
    while not q_list:
        j += 1
        if j == 100:
            os.system('clear')
            j = 0
            
        running = False
        try:
            Q = getJson('Q')
        except ValueError:
            continue
        config = getJson('config')
        N_GPUS = config['N_GPUS']
        print(end=f"\033[0;0f")
        if len(Q['done']) > 0:
            print("[Done]")
            print(f"id  {'Experiment':30}{'Status':9}{'Start-Time':17}{'End-Time':17}{'Duration':9}Command")
            for i, job in enumerate(Q['done']):
                s_time = job['s_time']
                e_time = job['e_time']
                duration = datetime.datetime.strptime(e_time, '%b-%d %H:%M:%S') - datetime.datetime.strptime(s_time, '%b-%d %H:%M:%S')
                duration_t = str(duration)
                print(f"{i+1:<4}{job['exp']:30}{job['status']:9}{s_time:17}{e_time:17}{duration_t:9}{job['cmd']}")
            print()
        
        print("[Running]")
        print(f"GPU {'Experiment':30}{'Start-Time':17}{'Elapsed':9}Command")
        free_gpu, gpu_utils = getFreeGPU(return_util=True)
        for i in range(N_GPUS):
            if i in free_gpu:
                print(f"{i:<4}{' '*56}IDLE")
            elif Q['now'][str(i)] is None:
                print(f"{i:<4}{' '*56}Unstationed Process")
            else:
                job = Q['now'][str(i)]
                now = datetime.datetime.now() + relativedelta(hours=9)
                now = datetime.datetime.strptime(now.strftime('%b-%d %H:%M:%S'), '%b-%d %H:%M:%S')
                s_time = job['s_time']
                elapsed = now - datetime.datetime.strptime(s_time, '%b-%d %H:%M:%S')
                elapsed_t = str(elapsed)
                print(f"{i:<4}{job['exp']:30}{s_time:17}{elapsed_t:9}{job['cmd']}")
                running = True
        print()
    
        if len(Q['queue']) > 0:
            print("[QUEUE]")
            print(f"id  {'Queue-Time':17}{'Experiment':30}Command")
            for i, job in enumerate(Q['queue']):
                print(f"{i+1:<4}{job['Qtime']:17}{job['exp']:30}{job['cmd']}")
            print()
                
        if running:
            print("[OUTPUT]")
            for i in range(N_GPUS):
                job = Q['now'][str(i)]
                if job is not None:
                    with open(f"{config['log_dir']}/{job['exp']}/nohup.out", 'r') as f:
                        lines = f.readlines()
                    if len(lines) > 3:
                        print(f"GPU: {i} | Util: {gpu_utils[i]:3}% | EXP: {job['exp']}")
                        # for line in reversed(lines):
                        #     if "Epoch" in line:
                        #         print(line)
                        #         break
                        for line in reversed(lines):
                            if "loss" in line:
                                print(line)
                                break
                        print("\n")
                
                # TODO
                # break
                
        if arg == 0:
            break
        
        time.sleep(arg)


def recv_data(sock, fname=None):
    BUFF_SIZE = 4096
    datalen = int.from_bytes(sock.recv(4), byteorder='big', signed=False)
    dtype = sock.recv(4).decode()
    data = b''
    while len(data) < datalen:
        partial_data = sock.recv(min(BUFF_SIZE, datalen - len(data)))
        data += partial_data
    if dtype == 'word':
        data = data.decode()
    elif dtype == 'file':
        with open(fname, 'wb') as f:
            f.write(data)
    return data

def send_data(sock, data, dtype):
    if dtype == 'word':
        databyte = data.encode()
    elif dtype == 'file':
        with open(data, 'rb') as f:
            databyte = f.read()
    try:
        sock.sendall(int(len(databyte)).to_bytes(length=4, byteorder='big', signed=False))
        sock.sendall(dtype.encode())
        sock.sendall(databyte)
    except:
        return False
    return True  

def log(arg):
    func, log_name = arg
    config = getJson('config')
    log_dir = config['log_dir']
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('1.233.226.232', 8082))

    try:
        send_data(sock, func, 'word')
        send_data(sock, log_name, 'word')
        
        if func == 'push':
            if recv_data(sock) == 'succ':
                log_top = os.path.join(log_dir, log_name)
                with zipfile.ZipFile('temp.zip', 'w', zipfile.ZIP_DEFLATED) as zipper:
                    for (path, dir, files) in os.walk(log_top):
                        for file in files:
                            if file.endswith('.v2'):
                                fname = os.path.join(path, file)
                                zipper.write(fname, os.path.relpath(fname, log_top))
                send_data(sock, 'temp.zip', 'file')
                os.remove('temp.zip')
            else:
                print(f"Your log {log_name} is already existing in server. Please use another.")
            
        elif func == 'pull':
            if recv_data(sock) == 'succ':
                recv_data(sock, fname='temp.zip')
                with zipfile.ZipFile('temp.zip', 'r', zipfile.ZIP_DEFLATED) as zipper:
                    zipper.extractall(os.path.join(log_dir, log_name))
                os.remove('temp.zip')
                
            else:
                print(f"Your log {log_name} doesn't exist in server. Please use another.")
            
        else:
            print("Not a proper subfunction for function log. use 'push' or 'pull'")
        
    except:
        pass
        
def tsb(port):
    os.system(f"tensorboard --logdir=logs --port {port} --host 0.0.0.0")

def getJson(name, root='.dh'):
    with open(f'{root}/{name}.json', 'r') as f:
        data = json.load(f)
    return data

def putJson(data, name, root='.dh'):
    with open(f'{root}/{name}.json', 'w') as f:
        json.dump(data, f, indent=4)
        
def defaultConfig(root='.dh'):
    config = {
        'refresh_time':10,
        'discord_url':"https://discordapp.com/api/webhooks/1237721827187626064/TmFZtEiIFHzVhCjk0jnSbR_ntsbqV6ZEuiN2JwxdvQ8v4XEu77_KCOvm8LXnRn8j3zbl",
        'used_mem_crit':1000,
        'see_type':'full',
        'N_GPUS':4,
        'cmd_exp_idx':2,
        'log_dir':'logs',
    }
    putJson(config, 'config')

def getFreeGPU(return_util=False):
    config = getJson('config')
    N_GPUS = config['N_GPUS']
    
    nvsmi = nvidia_smi.getInstance()
    res_util_list = nvsmi.DeviceQuery("utilization.gpu, memory.used")["gpu"]
    free_gpus = []
    gpu_utils = []
    for i in range(N_GPUS):
        mem_used = res_util_list[i]['fb_memory_usage']['used']
        gpu_util = res_util_list[i]['utilization']['gpu_util']
        if mem_used < config['used_mem_crit'] and gpu_util == 0:
            free_gpus.append(i)
        gpu_utils.append(gpu_util)
    if return_util:
        return free_gpus, gpu_utils
    return free_gpus
    

def isInitted():
    out = subprocess.run('ps -ef | grep "python scheduler.py initialize"', shell=True, capture_output=True)
    out = str(out.stdout, 'utf-8').strip().split('\n')
    if len(out) - 2 == 0:
        return 0
    else:
        for proc in out:
            if 'grep' not in proc:
                pid = int(proc.split()[1])
                return pid
            assert "something went wrong"

def main(args):
    args, unk = args
    if args.func not in ['init', 'exit', 'push', 'pop', 'see', 'console', 'clear', 'move', 'kill', 'log', 'tsb', 'initialize']:
        print("The subcommand should be on of 'init', 'exit', 'push', 'pop', 'see', 'clear', 'move', 'kill', 'log', 'tsb'. Use 'python schduler help' for help")
        return
    if isInitted() == 0 and args.func != 'init':
        print("The scheduler should be initialized first. Use 'python schduler init'. Moreover, use 'python schduler help' for help")
        return
    if args.func == 'see':
        if len(unk) == 0:
            arg = 0.01
        else:
            try:
                arg = float(unk[0])
            except ValueError:
                print("The first argument of function 'see' should be able to converted into float!")
                return
        print(arg)
    elif args.func == 'move':
        if len(unk) < 1:
            print("The function 'move' requires at least one arguments!")
            return
        elif len(unk) == 1:
            dest = 1
        else:
            try:
                dest = int(unk[1])
            except ValueError:
                print("The arguments of function 'move' should be able to converted into int!")
                return
        try:
            dep = int(unk[0])
        except ValueError:
            print("The arguments of function 'move' should be able to converted into int!")
            return
        arg = (dep, dest)
    elif args.func == 'pop':
        if len(unk) < 1:
            arg = 0
        if unk[0] == 'all':
            arg = 'all'
        else:
            try:
                arg = int(unk[0])
            except ValueError:
                print("The arguments of function 'pop' should be able to converted into int! (except 'all)")
                return
    elif args.func == 'clear':
        if len(unk) < 1:
            idx = 0
        else:
            try:
                idx = int(unk[0])
            except ValueError:
                print("The arguments of function 'clear' should be able to converted into int!")
                return
        arg = idx
    elif args.func == 'kill':
        if len(unk) < 1:
            print("GPU index for function 'kill' should be provided!")
        else:
            try:
                arg = int(unk[0])
            except ValueError:
                print("The arguments of function 'kill' should be able to converted into int!")
                return
    elif args.func == 'push':
        exp = None
        if unk[0] == '--exp':
            if len(unk) < 2:
                print("Experiment name for the argument '--exp' is missing!")
                return
            exp = unk[1]
            unk = unk[2:]
        if unk[0] == '--hour':
            if len(unk) < 2:
                print("Hour for the argument '--hour' is missing!")
                return
            try:
                hour = float(unk[1])
            except ValueError:
                print("Hour for the argument '--hour' in function 'push' should be able to converted into float!")
                return
            unk = unk[2:]
        else:
            hour = 0
            
        if len(unk) < 1:
            print("At least one command word is required for the function function 'push'!")
            return
        elif exp is None:
            try:
                exp_num = int(unk[0])
                unk = unk[1:]
            except ValueError:
                config = getJson('config')
                exp_num = config['cmd_exp_idx']
            if len(unk) < exp_num+1:
                print("The command is to short for the experiment index!")
                return
            exp = unk[exp_num]
        cmd = ' '.join(unk)
        arg = (exp, hour, cmd)
    elif args.func == 'log':
        arg = unk[:2]
    elif args.func == 'tsb':
        if len(unk) < 1:
            arg = 8080
        else:
            try:
                arg = int(unk[0])
            except ValueError:
                print("The arguments of function 'tsb' should be able to converted into int!")
                return
    else:
        arg = None
    globals()[args.func](arg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('func', type=str, help="Scheduler subfunction. Use 'python schduler help' for help")
    args = parser.parse_known_args()
    main(args)