#!/usr/bin/env python3
"""
打印合并 JSON 文件中每个 QA 条目的详细信息（彩色输出）。

Usage examples:
  python print_merged_details.py --file merged_1_gpt4omini.json --limit 3
  python print_merged_details.py --file merged_1_gpt4omini.json --query "Caroline" --limit 2
  python print_merged_details.py --file merged_1_gpt4omini.json --index 5

输出字段包括：evidence_sentence, initial_results, update_chain, speaker_1_memories, speaker_2_memories 等。
脚本会容错缺失字段并尝试常见变体（例如 spearker_memories / speaker_memories）。
"""
import argparse
import json
import os
import sys

COLORS = {
    'reset': '\033[0m',
    'question': '\033[95m',    # magenta
    'answer': '\033[94m',      # blue
    'evidence': '\033[92m',    # green
    'initial': '\033[93m',     # yellow
    'update': '\033[91m',      # red
    'speaker1': '\033[96m',    # cyan
    'speaker2': '\033[35m',    # purple
    'meta': '\033[90m',        # grey
}


def color(s, key):
    return f"{COLORS.get(key, '')}{s}{COLORS['reset']}"


def safe_get(d, *keys):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return None


def print_memory_entry(mem, prefix=''):
    # evidence_sentence
    evid = mem.get('evidence_sentence') or mem.get('evidence')
    if evid:
        print(prefix + color('evidence_sentence:', 'evidence'), evid)

    ts = mem.get('time_stamp') or mem.get('timestamp')
    if ts:
        print(prefix + color('time_stamp:', 'meta'), ts)

    # initial_results
    init = mem.get('initial_results')
    if init:
        print(prefix + color('initial_results:', 'initial'))
        for i, it in enumerate(init, 1):
            print(prefix + f"  [{i}] id={it.get('id', '')} event={it.get('event', '')}")
            if 'memory' in it:
                print(prefix + '    ' + color('memory:', 'initial') + ' ' + str(it['memory']))
            if 'previous_memory' in it:
                print(prefix + '    ' + color('previous_memory:', 'meta') + ' ' + str(it['previous_memory']))

    # update_chain
    upd = mem.get('update_chain')
    if upd:
        print(prefix + color('update_chain:', 'update'))
        for i, it in enumerate(upd, 1):
            print(prefix + f"  [{i}] id={it.get('id','')} event={it.get('event','')}")
            if 'memory' in it:
                print(prefix + '    ' + color('memory:', 'update') + ' ' + str(it['memory']))
            if 'previous_memory' in it:
                print(prefix + '    ' + color('previous_memory:', 'meta') + ' ' + str(it['previous_memory']))

    # other keys
    for k, v in mem.items():
        if k in ('evidence_sentence', 'evidence', 'time_stamp', 'timestamp', 'initial_results', 'update_chain'):
            continue
        # print small other fields
        if isinstance(v, (str, int, float)):
            print(prefix + color(f"{k}:", 'meta'), v)


def print_qa_entry(idx, item):
    header = f"[{idx}] {item.get('qa_question', '<no question>')}"
    print(color('=' * 80, 'meta'))
    print(color(header, 'question'))
    print(color('qa_answer:', 'answer'), item.get('qa_answer'))
    print(color('qa_response:', 'meta'), item.get('qa_response'))
    print(color('qa_category:', 'meta'), item.get('qa_category'))

    # person1/person2
    for p in ('person1', 'person2'):
        person = item.get(p)
        if not person:
            continue
        print(color(f"-- {p} name:", 'meta'), person.get('name'))
        mems = person.get('memories') or []
        if mems:
            for mi, mem in enumerate(mems, 1):
                print(color(f"  {p} memory #{mi}", 'meta'))
                print_memory_entry(mem, prefix='    ')
        else:
            print('    (no memories)')

    # speaker memories - try multiple possible keys
    spk1 = safe_get(item, 'speaker_1_memories', 'speaker1_memories', 'speaker_1', 'spearker_memories')
    spk2 = safe_get(item, 'speaker_2_memories', 'speaker2_memories', 'speaker_2')
    if spk1:
        print(color('-- speaker_1_memories:', 'speaker1'))
        for i, s in enumerate(spk1, 1):
            print(f"  [{i}] " + color(s.get('memory', str(s)), 'speaker1'))
            if 'timestamp' in s:
                print('     ' + color('timestamp:', 'meta') + ' ' + str(s['timestamp']))
            if 'score' in s:
                print('     ' + color('score:', 'meta') + ' ' + str(s['score']))

    if spk2:
        print(color('-- speaker_2_memories:', 'speaker2'))
        for i, s in enumerate(spk2, 1):
            print(f"  [{i}] " + color(s.get('memory', str(s)), 'speaker2'))
            if 'timestamp' in s:
                print('     ' + color('timestamp:', 'meta') + ' ' + str(s['timestamp']))
            if 'score' in s:
                print('     ' + color('score:', 'meta') + ' ' + str(s['score']))


def load_json_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def flatten_entries(data):
    # data might be mapping like {"0": [ ... ]} or a list
    entries = []
    if isinstance(data, dict):
        # iterate keys sorted for consistency
        for k in sorted(data.keys(), key=lambda x: int(x) if str(x).isdigit() else x):
            val = data[k]
            if isinstance(val, list):
                for it in val:
                    entries.append(it)
            else:
                entries.append(val)
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError('Unsupported JSON structure')
    return entries


def main():
    p = argparse.ArgumentParser(description='Print detailed QA entries from merged JSON with colors')
    p.add_argument('--file', '-f', default='merged_1_gpt4omini.json', help='路径到 JSON 文件 (相对 path 从当前目录)')
    p.add_argument('--index', '-i', type=int, help='按全局条目索引打印（从0开始）')
    p.add_argument('--query', '-q', help='按问题包含的子串进行过滤（不区分大小写）')
    p.add_argument('--limit', '-n', type=int, default=0, help='最多打印多少条（0 为无限）')
    args = p.parse_args()

    file_path = args.file
    if not os.path.isabs(file_path):
        # try relative to the script directory
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, file_path)

    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        sys.exit(2)

    data = load_json_file(file_path)
    entries = flatten_entries(data)

    # apply index
    if args.index is not None:
        if args.index < 0 or args.index >= len(entries):
            print('index 越界')
            sys.exit(2)
        to_print = [(args.index, entries[args.index])]
    else:
        # filter by query
        if args.query:
            q = args.query.lower()
            matched = [(i, e) for i, e in enumerate(entries) if q in str(e.get('qa_question','')).lower()]
        else:
            matched = list(enumerate(entries))

        if args.limit and args.limit > 0:
            to_print = matched[:args.limit]
        else:
            to_print = matched

    if not to_print:
        print('没有匹配的条目')
        return

    for idx, item in to_print:
        print_qa_entry(idx, item)


if __name__ == '__main__':
    main()
