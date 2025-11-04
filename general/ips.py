import socket
import psutil

def get_local_ips_with_psutil():
    # 使用 psutil 获取本机的 IP 地址。
    # :return: 本机 IP 地址列表
    ips = []
    try:
        # 获取所有网络接口
        addrs = psutil.net_if_addrs()
        for interface, addresses in addrs.items():
            for addr in addresses:
                if addr.family == socket.AF_INET and addr.address != '127.0.0.1':
                    ips.append(addr.address)
        return ips
    except Exception as e:
        print(f"获取 IP 地址失败: {e}")
        return []
    
def get_local_ip():
    # 获取本机的 IP 地址。
    # :return: 本机 IP 地址字符串
    try:
        # 获取本机的主机名
        hostname = socket.gethostname()
        # 通过主机名获取 IP 地址
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except Exception as e:
        return f"获取 IP 地址失败: {e}"
def is_localhost(hostname):
    # 判断给定的主机名或 IP 地址是否是本机回环地址。
    # :param hostname: 主机名或 IP 地址
    # :return: True 如果是回环地址，否则 False
    # 将主机名解析为 IP 地址
    try:
        ip_address = socket.gethostbyname(hostname)
    except socket.error:
        return False

    # 检查是否是回环地址
    return ip_address in ('127.0.0.1', '::1') or hostname.lower() == 'localhost'# or hostname.lower() == ip_address

def is_localip(ip):
    ips = get_local_ips_with_psutil()
    # 检查是否是回环地址
    return ip in ips
