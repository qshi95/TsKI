import pandas as pd
import numpy as np
import re

from table_processor import get_default_processor


TABLE_PROCESSOR = get_default_processor(max_cell_length=15, max_input_length=1024)


def process_table(table):
    header = list(table.columns)
    rows = np.array(table).tolist()
    table_content = {"header":header, "rows":rows}
    input_source, table_content = TABLE_PROCESSOR.process_pretraining_corpus(table_content, '', [])
    table_dict = dict()
    assert len(table_content['header']) == len(table_content['rows'][0])
    for i in range(len(table_content['header'])):
        table_dict[table_content['header'][i]] = [item[i] for item in table_content['rows']]

    return input_source.strip().lower(), pd.DataFrame(table_dict)
