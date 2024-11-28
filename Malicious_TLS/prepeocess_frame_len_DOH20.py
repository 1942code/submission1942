import dpkt
import os
import re
# print(os.environ['PATH'])
import socket
import argparse
import csv
import time
import subprocess
from scapy.all import *
from flowcontainer.extractor import extract
import pandas as pd

pd.set_option('display.max_rows', None)
import sys

csv.field_size_limit(sys.maxsize)

FLAGS = None
INPUT = "DOH21"
FILTER_LIST = None #[(["audio", "voip"], True), (["vpn", "tor"], False)]

PROTO_DICT = {6: "TCP", 17: "UDP"}
empty_pcap = []


def merge_csv_files(root_dir, output_file):
    # 创建一个空的列表来存储所有的行
    all_rows = []

    # 遍历根目录及其所有子目录
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # 如果文件是 CSV 文件
            if file.endswith('.csv'):
                with open(os.path.join(root, file), 'r') as f:
                    reader = csv.reader(f)
                    # 添加每一行到 all_rows
                    for row in reader:
                        all_rows.append(row)

    # 写入新的 CSV 文件
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in all_rows:
            writer.writerow(row)


def extract_name(pcap_path):
    base_name = os.path.basename(pcap_path)
    return(base_name.split('.')[0])


def get_files(dir_path):
    pcap_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if os.path.splitext(file)[-1] == '.pcap':
                pcap_files.append(os.path.join(root, file))
    return pcap_files


def pad_or_truncate(input_list, padding_length):
    if len(input_list) > padding_length:
        return input_list[:padding_length]
    else:
        return input_list + [0] * (padding_length - len(input_list))

def parse_pcap(pcap_path, file_name):
    # pcap_reader = PcapReader(pcap_path)
    # pcap_dict = {}
    # flow_first_src = {}  # 新增字典，用于存储每个五元组对应的流的第一个包的源地址
    result = extract(pcap_path, filter='(tcp or udp)')
    csv_file_path = os.path.splitext(pcap_path)[0] + ".csv"
    extract_count = 0
    discard_count = 0

    data = []
    for key in result: #key是(filename, procotol, stream_id)，每个key对应一个流
        value = result[key]

        payload_lengths = value.frame_lengths
        payload_lengths = pad_or_truncate(payload_lengths, 120)

        if 'DOH' in INPUT:
            if 'Benign' in pcap_path:
                label = 'benign'
                label1 = 'benign'
            elif 'dns2tcp' in pcap_path:
                label = 'dns2tcp'
                label1 = 'malicous'
            elif 'dnscat2' in pcap_path:
                label = 'dnscat2'
                label1 = 'malicous'
            elif 'iodine' in pcap_path:
                label = 'iodine'
                label1 = 'malicous'
            else:
                label = 'unknown'  # 如果没有匹配到，设置为 unknown 或其他默认值

        extract_count += 1
        data = payload_lengths + [label] + [label1]
        # print(row)

        df = pd.DataFrame([data])
        df.to_csv(csv_file_path, mode='a', index=False, header=False)
    print("Extracted flows: ", extract_count)
    # print("Discarded flows: ", discard_count)


def generic_parser(file_list):
    """Open up a pcap file and create a output file containing all one-directional parsed sessions"""
    for pcap_path in file_list:
        # try:
        print("Parsing " + pcap_path)
        parse_pcap(pcap_path, extract_name(pcap_path))
    for empty in empty_pcap:
        print(empty, '\n')

if __name__ == '__main__':
    pcap_list= get_files(INPUT)
    start_time = time.time()
    generic_parser(pcap_list)
    columns = [str(i) for i in range(120)] + ['label'] + ['label1']
    BASE_PATH = "/home/ucas/python_codes/dirpic/"
    output_filename = "DOH21.csv"
    merge_csv_files(BASE_PATH + INPUT, output_filename)
    df = pd.read_csv(output_filename)
    df.columns = columns
    df.to_csv(output_filename, index=None)

