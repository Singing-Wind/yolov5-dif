import os
import sys
import zipfile
import paramiko
from scp import SCPClient
try:
    from general.ips import is_localip
except:
    is_localip = None
import socket

def parse_ip_port(ips):
    if ':' in ips:
        ip, port = ips.split(':', 1)
        port = int(port)
    else:
        ip = ips
        port = 22
    return ip, port

def create_zip(zip_filename, base_dir,exclude_dirs=['__pycache__', 'runs', 'weights'],exclude_files=['Arial.ttf']):
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(base_dir):
            # 剔除不需要的文件夹
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                if file not in exclude_files:
                    if ('DOTA_devkit' not in root) or file=='image_split.py' or file=='DOTA_Dev2安装说明.txt':
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, base_dir)
                        zipf.write(file_path, arcname)

def safe_decode(byte_data):
    for encoding in ['utf-8', 'gbk', 'latin1']:
        try:
            return byte_data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return byte_data.decode('utf-8', errors='replace') 

def detect_remote_os(ssh):
    """判断远程系统是 Linux 还是 Windows"""
    try:
        stdin, stdout, stderr = ssh.exec_command("uname")
        uname_output = safe_decode(stdout.read()).strip().lower()
        if "linux" in uname_output or "darwin" in uname_output:
            return "linux"
        else:
            # 如果不是 Linux，可能是 Windows，尝试执行 Windows 命令作为二次确认
            stdin, stdout, stderr = ssh.exec_command("ver")
            ver_output = safe_decode(stdout.read()).strip().lower()
            if "windows" in ver_output:
                return "windows"
    except Exception as e:
        print(f"OS detection failed: {e}")
    return "unknown"

def create_remote_path(ssh, remote_path):
    is_windows = detect_remote_os(ssh) == 'windows'
    if is_windows:
        # 使用 PowerShell 检查目录是否存在
        path_str = f'"{remote_path}"'
        check_cmd = f'powershell -Command "if (Test-Path -Path {path_str} -PathType Container) {{ Write-Output exists }}"'
        # mkdir_cmd = f'powershell -Command "New-Item -ItemType Directory -Force -Path {path_str}"'
        mkdir_cmd = (
            f'powershell -NoProfile -Command "try {{ ' +
            f'New-Item -ItemType Directory -Force -Path {path_str} | Out-Null; Write-Output success ' +
            f'}} catch {{ Write-Output failed; $_.Exception.Message }}"'
        )
    else:
        # Linux/macOS 命令
        check_cmd = f'[ -d "{remote_path}" ] && echo "exists"'
        mkdir_cmd = f'mkdir -p "{remote_path}" && echo "success"'

    # 执行存在性检查
    stdin, stdout, stderr = ssh.exec_command(check_cmd)
    stdout.channel.recv_exit_status()
    output = safe_decode(stdout.read()).strip()

    if output != "exists":
        # 创建目录
        stdin, stdout, stderr = ssh.exec_command(mkdir_cmd)
        exit_status = stdout.channel.recv_exit_status()
        output = safe_decode(stdout.read()).strip()
        if output == 'success':
            parts = os.path.normpath(remote_path).split(os.path.sep)
            print(f"\033[31mCreated:{parts[-1]}\033[0m", end=' ')
        else:
            err = safe_decode(stderr.read()).strip()
            print(f"\033[31mFailed to create {remote_path}: {err}\033[0m")

def remote_path_exists(ssh, path, is_dir=False):
    flag = '-d' if is_dir else '-f'
    linux = path[0]=='/'
    if linux:
        stdin, stdout, stderr = ssh.exec_command(f'[ {flag} "{path}" ] && echo "exists"')
    else:
        stdin, stdout, stderr = ssh.exec_command(f'if exist "{path}" echo exists')
    exit_status = stdout.channel.recv_exit_status()
    output = stdout.read().decode().strip()
    return output == "exists"
def scp_transfer(ssh, local_file, remote_path, machine):
    name = os.path.basename(local_file)
    # if remote_base_path==os.path.dirname(local_file): #local scp
    #     name = add_suffix_to_filename(name,'_scp')
    remote_file = os.path.join(remote_path,name).replace('\\', '/')
    # 检查远程文件是否存在
    if not remote_path_exists(ssh, remote_file, is_dir=os.path.isdir(local_file)):
        # 使用SCP传输文件
        try:
            with SCPClient(ssh.get_transport()) as scp:
                scp.put(local_file, remote_path, recursive=True)
            # print(f"\033[32m{os.path.basename(local_file)}->\033[32m{remote_path}\033[0m")
            print(f"\033[32m->\033[32m{remote_path}\033[0m")
        except Exception as e:
            print(f'\033[31mssh trans failure: {e}\033[0m')
    else:
        # 文件存在，红色报警并暂停确认
        print(f"\033[31m警告：远程文件已存在:{remote_file}\033[0m 取消")
        # exit(0)
        # confirm = input("是否覆盖？(y/n): ").strip().lower()
        # if confirm != 'y':
        #     print("传输已取消")

def connect_with_timeout(ssh, machine, timeout=8):
    # 尝试连接 SSH 服务器，设置超时时间。
    # :param machine: 包含连接信息的字典（ip, port, user, password）
    # :param timeout: 连接超时时间（秒）
    # :return: True 如果连接成功，否则 False
    # ssh = paramiko.SSHClient()
        # 尝试连接，设置超时时间
    machine['ip'] = [machine['ip']] if isinstance(machine['ip'], str) else machine['ip']
    for ip in machine['ip']:
        # ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            ssh.connect(
                ip,
                port=machine['port'],
                username=machine['user'],
                password=machine['password'],
                timeout=timeout
            )
            # print(f"成功连接到 {ip}:{machine['port']}")
            return True
        except socket.timeout:
            print(f"\033[31m连接超时：{ip}:{machine['port']} 在 {timeout} 秒内无响应\033[0m",end='')
        except paramiko.ssh_exception.NoValidConnectionsError:
            print(f"\033[31m,无法连接到 {ip}:{machine['port']}，请检查网络或主机状态\033[0m",end='')
        except paramiko.ssh_exception.AuthenticationException:
            print(f"\033[31m认证失败：用户名或密码错误\033[0m",end='')
        except Exception as e:
            print(f"\033[31m连接失败：{e}\033[0m",end='')
        finally:
            # 关闭连接
            ssh.close()
    return False

def backup_to_remote(machine, local_zip, remote_base_path,timeout=8):#ip, user, password
    # remote_path = os.path.join(remote_base_path, os.path.basename(local_zip))
    # 连接到远程机器
    print('linking..')
    status = False
    machine['ip'] = [machine['ip']] if isinstance(machine['ip'], str) else machine['ip']
    ports = machine.get('port', 22)
    for i, ip in enumerate(machine['ip']):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        if isinstance(ports, list):
            machine['port'] = ports[i]
        else:
            machine['port'] = ports
        try:
            ssh.connect(ip, port=machine['port'], username=machine['user'], password=machine['password'],timeout=timeout)
            status = True
            print(f"\t\033[32m{ip}:{machine['port']} ok\033[0m", end=' ')
        except socket.timeout:
            print(f"\t\033[33m连接超时：{ip}:{machine['port']} 在 {timeout} 秒内无响应\033[0m",end='')
        except paramiko.ssh_exception.NoValidConnectionsError:
            print(f"\t\033[33m,无法连接到 {ip}:{machine['port']}，请检查网络或主机状态\033[0m",end='')
        except paramiko.ssh_exception.AuthenticationException:
            print(f"\t\033[33m认证失败：用户名或密码错误\033[0m",end='')
        except Exception as e:
            print(f"\t\033[33m连接失败：{e}\033[0m",end='')
        finally:
            # 关闭连接
            # ssh.close()
            # print('\033[31mfail\033[0m')
            pass
        
        if status:
            # 创建远程路径
            if is_localip is not None:
                if is_localip(ip): #in one local net
                    assert os.name=='nt' or  remote_base_path==os.path.dirname(local_zip)
            create_remote_path(ssh, remote_base_path)

            # 传输文件
            scp_transfer(ssh, local_zip, remote_base_path, machine)
            
        else:
            print('\033[31mfail\033[0m')
        
        ssh.close() 
        if status:
            return True

if __name__ == "__main__":
    local_zip_path = '' #'/media/liu/a2254a68-9f90-4b44-ab2a-ffc55b361238/datas/coco128/names_vec.npz'
    if not os.path.exists(local_zip_path):
        # 获取当前 .py 文件所在的绝对路径
        current_file_path = os.path.abspath(__file__)
        # 获取当前工作目录的上一级文件夹
        base_dir = os.path.dirname(current_file_path) #os.getcwd()
        # 设置压缩包的文件名和路径
        local_path = os.path.dirname(base_dir)
        local_zip_path = os.path.join(local_path, 'yolov5-dif-WHU-hull_ft-rename_exp.zip')
        #
        # 剔除的文件夹列表
        exclude_dirs = ['__pycache__', 'runs', 'weights', 'checkpoint', 'ckpt']
        # 剔除的文件列表
        exclude_files = ['Arial.ttf']
        create_zip(local_zip_path, base_dir,exclude_dirs=exclude_dirs,exclude_files=exclude_files)
    else:
        local_path = os.path.dirname(local_zip_path)

    # 去掉前两级目录
    username = os.getlogin()
    print(f"本地{username}文件:\033[33mfile:{local_zip_path}\033[0m")
    # 使用 os.path.sep 分割路径，然后从第三部分开始拼接
    parts = local_zip_path.split(os.path.sep)
    if (os.name=='posix' and parts[0]=='' and parts[2]==username) or (os.name=='nt' and ':' in parts[0]):
        middle_path = os.path.sep.join(parts[3:-1]) if os.name=='posix' else os.path.sep.join(parts[parts.index('workspace'):-1]).replace("\\", "/") # 去掉前两级和最后一级
        
        # 远程备份
        remote_machines = [
            {'name':'4090','ip':['192.168.110.45','10.10.10.3','10.5.0.1'], 'user':'liu', 'password':'liuzyn', 'datas':'/media/liu/088d8f6e-fca3-4aed-871f-243ad962413b'}, #4090
            {'name':'4090-2','ip':['192.168.110.48','10.10.10.4','10.5.0.2'], 'user':'liu', 'password':'liuzyn', 'datas':'/media/liu/f4854541-32b0-4d00-84a6-13d3a5dd30f2'}, #4090-2
            {'name':'4090x2svr','ip':['192.168.210.171','10.10.10.5','10.5.0.3'], 'user':'user', 'password':'liuzyn', 'datas':'/data'}, #4090x2 svr
            {'name':'zh3090-277','ip':['192.168.31.140','10.10.10.9','10.5.0.8'], 'user':'liu', 'password':'liuzyn', 'datas':'/media/liu/a2254a68-9f90-4b44-ab2a-ffc55b361238'}, #zh3090
            {'name':'darknet-524','ip':['192.168.31.240','10.10.10.7','10.5.0.5'], 'user':'liu', 'password':'liuzyn', 'datas':'/home/liu/data/home/liu/workspace/darknet'}, #darknet-524
            {'name':'709-4090','ip':['192.168.104.166','10.10.10.6','10.5.0.4'], 'user':'liu', 'password':'liuzyn', 'datas':'/media/data4T'}, #709-4090
            {'name':'office3060','ip':['192.168.110.29','10.10.10.8','10.5.0.6'], 'user':'liu', 'password':'liuzyn', 'datas':'/media/liu/124CDDC24CDDA0B1'}, #3060
            {'name':'g1','ip':['10.10.10.2', '192.168.34.169'], 'user':'liujin', 'password':'Liu12345.',  'root':'data', 'datas':'/data/liujin/workspace'}, #A100-g1
            {'name':'sd2','ip':['59.172.178.26:7922', '192.168.34.79'], 'user':'liujin', 'password':'Liu12345.', 'root':'sgg', 'datas':'/sgg/liujin/workspace'}, #A100-sd2
            {'name':'win-main','ip':['192.168.31.104','10.10.10.10','10.5.0.7'], 'user':'liujin', 'password':'liuzyn', 'path':'D:/Lab/darknet-vs2013-A2271-427/workspace', 'datas':'E:/datas'}, #main windows
        ]
        
        for machine in remote_machines:
            machine_user = f"{machine['name']}:{machine['user']}@{machine['ip']}"
            ips,ports=[],[]
            for ip_port in machine['ip']:
                ip,port = parse_ip_port(ip_port)
                ips.append(ip)
                ports.append(port)
            machine['ip'] = ips
            machine['port'] = ports

            print(machine_user.ljust(34), end=' ')
            if 'datas' not in parts:
                if 'win'!=machine['name'][:3]:
                    # root = machine.get('root',parts[1] if os.name=='posix' else 'home')
                    root = machine.get('root', 'home')
                    remote_base_path = f'/{root}/{machine["user"]}/{middle_path}'
                else:#windows
                    start_path_idx = 4 if os.name=='posix' else 2
                    local_path2 = os.path.sep.join(parts[start_path_idx:-1]).replace('\\', '/')
                    remote_base_path = f'{machine["path"]}/{local_path2}'
            else:
                idx = parts.index('datas')
                new_parts = [machine["datas"]] + parts[idx:-1]  # 替换前缀部分
                remote_base_path = os.path.join(*new_parts)
            if is_localip is None or not is_localip(machine['ip']):
                backup_to_remote(machine, local_zip_path, remote_base_path, timeout=4)
            else:#local ip machine
                assert remote_base_path==local_path
                # remote_base_path = add_suffix_to_filename(remote_base_path,'_scp')
                print(f'\033[92m就在本机{machine_user}{local_path}\033[0m 不用拷贝.')
            
    else:
        print(f'parts[0]={parts[0]} parts[2]={parts[2]}')
        print(f'parts={parts}')