# encoding=utf8
import json
import sys
sys.path.append('..')
# from execution.execute import get_sub_program
import pandas as pd
import numpy
from beam_search import dynamic_programming
from multiprocessing import Pool
import multiprocessing
import sys
import time
import argparse
import os
import re
import numpy as np
import random
random.seed(2)
from random import sample
from utils import TABLE_PROCESSOR
import jsonlines

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="../corpus", help="which folder to store the results")
parser.add_argument("--table_id_file", type=str, default="./table_id.txt", help="table id file")
parser.add_argument("--table_source_folder", type=str, default="../all_csv/", help="table id file")
args = parser.parse_args()

fw = jsonlines.open(os.path.join(args.output, 'pretraining_corpus_evidence_3000000.jsonl'), 'w')

def isnumber(string):
    return string in [numpy.dtype('int64'), numpy.dtype('int32'), numpy.dtype('float32'), numpy.dtype('float64')]


def list2tuple(inputs):
    mem = []
    for s in inputs:
        mem.append(tuple(s))
    return mem

def split(string, option):
    if option == "row":
        return string.split(',')[0]
    else:
        return string.split(',')[1]

def process_program(program):
    return program.replace('{', ' { ').replace('}', ' } ').replace(';', ' ; ').replace('  ', ' ').strip()

def process_table(table):
    header = list(table.columns)
    rows = np.array(table).tolist()
    table_content = {"header":header, "rows":rows}
    _, table_content = TABLE_PROCESSOR.process_pretraining_corpus(table_content, '', [])
    table_dict = dict()
    assert len(table_content['header']) == len(table_content['rows'][0])
    for i in range(len(table_content['header'])):
        table_dict[table_content['header'][i]] = [item[i] for item in table_content['rows']]
    
    # table_text = [header] + rows
    table_text = []
    # table_text = [temp for item in table_text for temp in item]
    for item in [header] + rows:
        temp_list = []
        for temp in item:
            temp_list.append(str(temp))
        table_text.append(temp_list)

    return table_text, pd.DataFrame(table_dict)

def func(inputs):
    index, table_name = inputs
    program_list = set()
    # t = pd.read_csv('../table_source/all_csv/{}'.format(table_name), delimiter="#", encoding='utf-8')
    t = pd.read_csv(args.table_source_folder + table_name, delimiter="#", encoding='utf-8')
    processed_table, t = process_table(t)
    t.fillna('')
    mapping = {i: "num" if isnumber(t) else "str" for i, t in enumerate(t.dtypes)}
    t_columns = [column for column in t]

    # 随机选 str横纵坐标，num横纵坐标
    str_columns = [key for key in mapping if mapping[key] == 'str']
    num_columns = [key for key in mapping if mapping[key] == 'num']
    str_tuples = [(i,j) for i in range(len(t)) for j in str_columns]
    num_tuples = [(i,j) for i in range(len(t)) for j in num_columns]
    if len(num_tuples) != 0 and len(str_tuples) != 0:   # 因为表格多 所以我们只选这一种的
        table_id_list.append(table_name)
        total_tuples = [(i,j) for i in str_tuples for j in num_tuples]
        random.shuffle(total_tuples)
        for tuple in total_tuples:
            str_t, num_t = tuple
            mem_str = [(t_columns[str_t[1]], t.iloc[str_t[0], str_t[1]])]
            head_str = [item[0] for item in mem_str]
            mem_num = [(t_columns[num_t[1]], int(t.iloc[num_t[0], num_t[1]]))]
            head_num = [item[0] for item in mem_num]
            res = dynamic_programming(table_name, t, mem_str, mem_num, head_str, head_num, 5)
            true_returned_list = [item.split('=')[0] + '= 1' for item in res[1] if item.split('=')[1] == 'True']
            false_returned_list = [item.split('=')[0] + '= 0' for item in res[1] if item.split('=')[1] == 'False']
            
            if len(true_returned_list) > len(false_returned_list):
                true_returned_list = random.sample(true_returned_list, len(false_returned_list))
            else:
                false_returned_list = random.sample(false_returned_list, len(true_returned_list))

            program_list.update(true_returned_list+false_returned_list)
            # print(len(program_list))

            if len(program_list) > 1000:
                break

        for program in program_list:
            row = dict()
            row['table_text'] = processed_table
            row['evidence'] = process_program(program.strip()).split('=')[0].strip()
            row['label'] = int(process_program(program.strip()).split('=')[1].strip()) 
            fw.write(row)
    
    print('Finished ' + str(index) + ' Tables.')

if __name__ == '__main__':

    with open(args.table_id_file, 'r') as f:
        table_id_list = [item.strip() for item in f.readlines()]

    index_list = list(range(len(table_id_list)))
    print(index_list)
    exit()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    for arg in zip(index_list, table_id_list):
        func(arg)

    # cores = multiprocessing.cpu_count()
    # print("Using {} cores".format(cores))
    # pool = Pool(cores)
    # res = pool.map(func, zip(index_list, table_id_list))

    # pool.close()
    # pool.join()