"""Offline task compilation for bounded integer-polynomial program synthesis.

The built-in distribution is synthetic. External DeepMind question/answer files
can be converted with scripts/prepare_program_tasks.py; no download is implicit.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
import numpy as np

OPS = ('add','subtract','multiply','divide_exact')


def polynomial_op(op, a, b, degree, limit):
    a=np.asarray(a,dtype=np.int64);b=np.asarray(b,dtype=np.int64)
    if op==0: result=a+b
    elif op==1: result=a-b
    elif op==2:
        full=np.convolve(a,b)
        if np.any(full[degree+1:]):raise ValueError('polynomial degree exceeds bound')
        result=full[:degree+1]
    else:
        if b[0]==0 or np.any(b[1:]) or np.any(a % b[0]):
            raise ValueError('division must have an exactly dividing scalar denominator')
        result=a//b[0]
    if np.any(np.abs(result)>limit):raise ValueError('coefficient exceeds bound')
    return tuple(int(x) for x in result)


def compile_task(expression, degree=3, limit=1024, max_inputs=6, max_ops=4, answer=None, source='synthetic'):
    tree=ast.parse(expression.replace('^','**'),mode='eval').body
    leaves=[]
    def scalar(n):
        if type(n) is not int or abs(n)>limit:raise ValueError('only bounded integer constants are supported')
        return (n,)+(0,)*degree
    def leaf(node):
        if isinstance(node,ast.Constant):return scalar(node.value)
        if isinstance(node,ast.Name) and node.id=='x':
            if degree<1:raise ValueError('x requires polynomial degree >=1')
            return (0,1)+(0,)*(degree-1)
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub) and isinstance(node.operand,ast.Constant) and type(node.operand.value) is int:
            return scalar(-node.operand.value)
        return None
    # Expand powers into ordinary multiplications, with reusable subexpressions.
    def expand(node):
        if leaf(node) is not None:return node
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):
            return ast.BinOp(left=ast.Constant(0),op=ast.Sub(),right=expand(node.operand))
        if not isinstance(node,ast.BinOp):raise ValueError('unsupported expression syntax')
        if isinstance(node.op,ast.Pow):
            if not isinstance(node.right,ast.Constant) or type(node.right.value) is not int or not 0<=node.right.value<=8:
                raise ValueError('exponent must be an integer from 0 to 8')
            n=node.right.value
            if n==0:return ast.Constant(1)
            result=expand(node.left)
            for _ in range(n-1):result=ast.BinOp(left=result,op=ast.Mult(),right=expand(node.left))
            return result
        if not isinstance(node.op,(ast.Add,ast.Sub,ast.Mult,ast.Div)):raise ValueError('unsupported operator')
        return ast.BinOp(left=expand(node.left),op=node.op,right=expand(node.right))
    tree=expand(tree)
    def collect(node):
        value=leaf(node)
        if value is not None:
            if value not in leaves:leaves.append(value)
        else:collect(node.left);collect(node.right)
    collect(tree)
    if len(leaves)>max_inputs:raise ValueError('too many distinct inputs')
    plan=[];values=list(leaves);cache={}
    def build(node):
        value=leaf(node)
        if value is not None:return leaves.index(value)
        key=ast.dump(node)
        if key in cache:return cache[key]
        i,j=build(node.left),build(node.right)
        op={ast.Add:0,ast.Sub:1,ast.Mult:2,ast.Div:3}[type(node.op)]
        result=polynomial_op(op,values[i],values[j],degree,limit)
        plan.append((op,i,j));values.append(result)
        if len(plan)>max_ops:raise ValueError('reference program exceeds operation bound')
        cache[key]=len(values)-1
        return cache[key]
    root=build(tree);target=values[root]
    if answer is not None:
        gold=compile_task(str(answer),degree,limit,max_inputs=32,max_ops=128)['target']
        if tuple(gold)!=target:raise ValueError('provided answer does not match exact expression value')
    canonical=ast.dump(tree)
    fingerprint=hashlib.sha256(canonical.encode()).hexdigest()
    # Token grammar is deliberately explicit; signs and whole integer literals
    # are separate from operation/action IDs. Parentheses are retained.
    pretty=ast.unparse(ast.fix_missing_locations(tree))
    tokens=re.findall(r'\d+|[x()+*/\-]',pretty)
    return dict(expression=pretty,inputs=[list(x) for x in leaves],plan=[list(x) for x in plan],
                target=list(target),program_values=[list(x) for x in values],tokens=tokens,
                fingerprint=fingerprint,source=source,reference_ops=len(plan))


def task_split(fingerprint):
    return 'test' if int(fingerprint[:8],16)%5==0 else 'train'


def make_tasks(count,seed,min_ops,max_ops,degree,limit,max_inputs,split='train',problem_file=None):
    if problem_file:
        records=[json.loads(line) for line in Path(problem_file).read_text().splitlines() if line.strip()]
        candidates=[]
        for row in records:
            # Recompile, independently checking both cached targets and bounds.
            task=compile_task(row['expression'],degree,limit,max_inputs,max_ops,
                              answer=row.get('answer'),source=row.get('source','external'))
            if 'target' in row and list(row['target'])!=task['target']:raise ValueError('cached task target mismatch')
            candidates.append(task)
        np.random.default_rng(seed).shuffle(candidates)
    else:
        rng=np.random.default_rng(seed)
        def expression(n):
            if n==0:
                return 'x' if degree>0 and rng.random()<.25 else str(int(rng.choice([-5,-4,-3,-2,2,3,4,5])))
            left=int(rng.integers(n));right=n-1-left
            return f'({expression(left)}{rng.choice(["+","-","*"])}{expression(right)})'
        def generated():
            for _ in range(max(10000,count*2000)):
                try:yield compile_task(expression(int(rng.integers(min_ops,max_ops+1))),degree,limit,max_inputs,max_ops)
                except ValueError:continue
        candidates=generated()
    selected=[];seen=set()
    for task in candidates:
        if task['fingerprint'] in seen or task_split(task['fingerprint'])!=split:continue
        if not min_ops<=task['reference_ops']<=max_ops:continue
        if task['target'] in task['inputs']:continue  # Remove immediate-submit tasks.
        if len(task['inputs'])+task['reference_ops']>max_inputs+max_ops:continue
        seen.add(task['fingerprint']);selected.append(task)
        if len(selected)==count:return selected
    raise ValueError(f'Only {len(selected)} compatible unique {split} tasks; requested {count}. Increase bounds/data.')


PROGRAM_DEFAULTS=dict(program_min_ops=2,program_max_ops=4,program_degree=3,
                      program_max_inputs=6,program_registers=10,program_coefficient_limit=1024,
                      program_teacher_probability=.75,program_problem_file=None)


PROGRAM_EVAL_DEFAULTS=dict(program_eval_k=128,program_eval_actions=4,program_eval_prefixes=2)


def add_program_arguments(parser):
    for name,default in {**PROGRAM_DEFAULTS,**PROGRAM_EVAL_DEFAULTS}.items():
        parser.add_argument('--'+name,type=(str if default is None else type(default)),default=default)


def program_kwargs(args):
    return {name:getattr(args,name) for name in PROGRAM_DEFAULTS}
