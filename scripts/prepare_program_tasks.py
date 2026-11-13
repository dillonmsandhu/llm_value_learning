#!/usr/bin/env python3
"""Convert an existing DeepMind arithmetic/polynomial Q/A file to validated JSONL.

No network or SymPy dependency. Supported questions start Calculate, Expand,
Simplify or Evaluate, followed by an expression over integer constants and x.
Rejects unsupported language/rationals/domains instead of approximating answers.
"""
import argparse
import ast
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from envs.program_tasks import compile_task


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--degree',type=int,default=3);p.add_argument('--max_ops',type=int,default=4)
    p.add_argument('--max_inputs',type=int,default=6);p.add_argument('--coefficient_limit',type=int,default=1024)
    args=p.parse_args()
    destination=Path(args.output)
    if destination.exists():p.error('output already exists')
    lines=Path(args.input).read_text().splitlines()
    if len(lines)%2:p.error('expected alternating question and answer lines')
    accepted=[];rejected=[];seen=set()
    for index in range(0,len(lines),2):
        question,answer=lines[index].strip(),lines[index+1].strip()
        try:
            match=re.fullmatch(r'(?:Calculate|Expand|Simplify|Evaluate)\s+(.+?)[.?]?',question)
            if not match:raise ValueError('unsupported question template')
            expression=match.group(1).rstrip('.?')
            variables={n.id for n in ast.walk(ast.parse(expression.replace('^','**'),mode='eval')) if isinstance(n,ast.Name)}
            if len(variables)==1:
                variable=next(iter(variables))
                expression=re.sub(r'\b'+re.escape(variable)+r'\b','x',expression)
                answer=re.sub(r'\b'+re.escape(variable)+r'\b','x',answer)
            task=compile_task(expression,args.degree,args.coefficient_limit,args.max_inputs,args.max_ops,
                              answer=answer,source='DeepMind Mathematics Dataset/'+Path(args.input).stem)
            if task['fingerprint'] in seen:raise ValueError('duplicate expression')
            seen.add(task['fingerprint'])
            task.update(question=question,answer=answer,source_index=index//2)
            accepted.append(task)
        except (ValueError,SyntaxError,TypeError,RecursionError) as exc:
            rejected.append(dict(source_index=index//2,reason=str(exc)))
    report=dict(input=str(Path(args.input).resolve()),accepted=len(accepted),rejected=len(rejected),rejections=rejected)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.with_suffix(destination.suffix+'.report.json').write_text(json.dumps(report,indent=2)+'\n')
    if not accepted:raise SystemExit('No compatible records; see rejection report. No task file written.')
    destination.write_text(''.join(json.dumps(t,sort_keys=True)+'\n' for t in accepted))
    print(f'Wrote {len(accepted)} validated tasks; rejected {len(rejected)}. Report: {destination}.report.json')


if __name__=='__main__':main()
