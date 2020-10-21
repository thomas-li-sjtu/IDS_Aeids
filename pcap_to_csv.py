from scapy.all import PcapReader
import numpy


with open('input.csv', 'a') as f:
    f.write('时间戳,协议,源ip,源port,目的ip,目的port,真实标签,\n')

s1 = PcapReader("ctu13_12-1.pcap")
infected_ip = ['147.32.84.165', '147.32.84.191', '147.32.84.192']

num = 0
index = 0
try:
    while True:
        data = s1.read_packet()
        if data[1].name == 'ARP':
            src = data[1].psrc
            dst = data[1].pdst
            sport = None
            dport = None
            protocol = data[1].name
            time = data[0].time
            origin = data[0].original
            if time > 1313743991 and src in infected_ip:
                label = 1
            else:
                label = 0
        elif data[2].name == 'UDP':
            # 从IP层拿到ip地址
            src = data[1].src
            dst = data[1].dst
            # 从UDP层拿到端口号
            sport = data[2].sport
            dport = data[2].dport
            protocol = data[2].name
            time = data[0].time
            origin = data[0].original
            if time > 1313743991 and src in infected_ip:
                label = 1
            else:
                label = 0
        elif data[2].name == 'TCP':
            # 从IP层拿到ip地址
            src = data[1].src
            dst = data[1].dst
            # 从TCP层拿到端口号
            sport = data[2].sport
            dport = data[2].dport
            protocol = data[2].name
            time = data[0].time
            origin = data[0].original
            if time > 1313743991 and src in infected_ip:
                label = 1
            else:
                label = 0
        elif data[2].name == 'ICMP':
            # 从IP层拿到ip地址
            src = data[1].src
            dst = data[1].dst
            sport = None
            dport = None
            protocol = data[2].name
            time = data[0].time
            origin = data[0].original
            if time > 1313743991 and src in infected_ip:
                label = 1
            else:
                label = 0
        else:
            continue
        # 对origin进行处理
        num += 1
        with open('input.csv', 'a') as f:
            f.write(str(time) + ',' + str(protocol) + ',' + str(src) + ',' + str(sport) + ',' + str(dst) + ',' + str(
                dport) + ',' + str(label) + '\n')
        if num % 1000 == 0:
            print(num)
except EOFError:
    s1.close()
